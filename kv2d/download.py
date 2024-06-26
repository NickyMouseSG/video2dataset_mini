from concurrent.futures import (
    ThreadPoolExecutor,
    wait,
    FIRST_COMPLETED,
    ProcessPoolExecutor,
)
from threading import Semaphore
from loguru import logger
import tempfile
from dataclasses import dataclass
from typing import List
import os, os.path as osp
import pathos.multiprocessing as mp
import multiprocessing
import pandas as pd
import time
from huggingface_hub import HfApi
from torchvision import transforms as IT
from PIL import Image
import io
import numpy as np
import torch
import sys
from einops import rearrange
import yt_dlp
from ffmpy import FFmpeg, FFprobe
from tqdm import tqdm
from queue import Queue
from glob import glob

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from .writer import BufferedTextWriter, get_writer, UploadArgs
from .sharder import Sharder
from .utils import shardid2name, safe_open
from .process import ImageProcessArgs, get_processor

from kn_util.data.video import download_youtube_as_bytes, read_frames_decord
from kn_util.data.video.download import StorageLogger
from kn_util.utils.rich import get_rich_progress_mofn
from kn_util.utils.download import MultiThreadDownloaderInMem, CoroutineDownloaderInMem
from kn_util.utils.error import SuppressStdoutStderr
from kn_util.utils.system import run_cmd, force_delete_dir, clear_process_dir
from kn_util.tools.lfs import upload_files


def _download_single_youtube(url, size=256, timestamp=None):
    # use a fake logger to suppress output and capture error
    storage_logger = StorageLogger()

    # default video format from video2dataset
    # https://github.com/iejMac/video2dataset/blob/main/video2dataset/data_reader.py
    video_format = f"wv*[height>={size}][ext=mp4]/" f"w[height>={size}][ext=mp4]/" "bv/b[ext=mp4]"
    video_bytes, errorcode = download_youtube_as_bytes(
        url,
        video_format=video_format,
        logger=storage_logger,
        timestamp=timestamp,
    )

    return video_bytes, errorcode, storage_logger.storage["error"]


def _download_single_direct(url):
    try:
        downloader = CoroutineDownloaderInMem(verbose=False, max_retries=0)
        video_bytes = downloader.download(url)
    except Exception as e:
        return None, 1, str(e)

    return video_bytes, 0, None


def _get_online_video_size(url):
    # ffmpeg -i <url> -v quiet -select_streams v:0 -show_entries stream=width,height -of csv=p=0
    ffp = FFprobe(
        inputs={url: None},
        global_options="-v quiet -select_streams v:0 -show_entries stream=width,height -of csv=p=0",
    )
    cmd = ffp.cmd
    result_str = run_cmd(cmd, verbose=False).stdout

    result = result_str.strip().split(",")
    try:
        result = [int(_) for _ in result]
    except:
        raise ValueError(f"Failed to get video size from {url}\n{cmd}")

    return result


def _get_target_size(orig_size, size, mode="shortest", divisible_by=1):
    cmp = lambda x, y: x < y if mode == "shortest" else x > y
    if cmp(orig_size[0], orig_size[1]):
        target_size = [size, int(size * orig_size[1] / orig_size[0])]
    else:
        target_size = [int(size * orig_size[0] / orig_size[1]), size]

    target_size[0] = target_size[0] // divisible_by * divisible_by
    target_size[1] = target_size[1] // divisible_by * divisible_by

    return target_size


def _download_single_ffmpeg(url, timestamp=None):
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            output_kwargs = ["-c copy"]
            if timestamp is not None:
                # st, ed = parse_timestamps(timestamp)
                st, ed = timestamp.split("-")
                output_kwargs += [f"-ss {st} -to {ed}"]

            output_kwargs = " ".join(output_kwargs)

            global_options = "-hide_banner -loglevel error -y "
            global_options += " -http_persistent 1"

            ff = FFmpeg(
                inputs={url: None},
                outputs={f.name: output_kwargs},
                global_options=global_options,
            )

            cmd = ff.cmd
            run_cmd(cmd, verbose=False)

            with open(f.name, "rb") as f:
                video_bytes = f.read()
                return video_bytes, 0, None
    except Exception as e:
        return None, 1, str(e)


def download_single(url, size, semaphore, timestamp=None, media="video"):
    _bytes = None
    errorcode = 0
    error_str = None
    semaphore.acquire()

    try:
        if media == "video":
            if not (url.endswith(".mp4") or url.endswith(".avi") or url.endswith(".mkv")):
                _bytes, errorcode, error_str = _download_single_youtube(url, size=size, timestamp=timestamp)
            else:
                _bytes, errorcode, error_str = _download_single_direct(url)

        elif media == "image":
            _bytes, errorcode, error_str = _download_single_direct(url)
        else:
            raise ValueError(f"Invalid media type: {media}")
    except Exception as e:
        return None, 1, str(e)

    return _bytes, errorcode, error_str


def download_generator(
    url_shard,
    id_shard,
    timestamp_shard,
    meta_shard,
    # download args
    size=360,
    media="video",
    # parallel
    num_threads=16,
    semaphore_limit=32,
    max_retries=3,
):
    executor = ThreadPoolExecutor(num_threads)
    semaphore = Semaphore(semaphore_limit)

    def submit_job(url, _id, timestamp, meta, retry_cnt=0):
        future = executor.submit(
            download_single,
            url=url,
            size=size,
            timestamp=timestamp,
            media=media,
            semaphore=semaphore,
        )
        future._input = dict(meta=meta, id=_id, url=url, timestamp=timestamp, retry_cnt=retry_cnt)
        return future

    def submit_jobs(ids):
        futures = set()
        for i in ids:
            future = submit_job(url_shard[i], id_shard[i], timestamp_shard[i], meta_shard[i], retry_cnt=0)
            futures.add(future)
        return futures

    # first submit
    not_done = submit_jobs(range(len(url_shard)))

    # polling
    while len(not_done) > 0:
        done, not_done = wait(not_done, return_when=FIRST_COMPLETED, timeout=1.0)
        for future in done:
            video_bytes, errorcode, error = future.result()
            url = future._input["url"]
            meta = future._input["meta"]
            _id = future._input["id"]
            timestamp = future._input["timestamp"]
            retry_cnt = future._input["retry_cnt"]

            if errorcode == 0:
                yield _id, video_bytes, errorcode
            else:
                yield _id, None, errorcode
                if retry_cnt + 1 < max_retries:
                    submit_job(url, _id, timestamp, meta, retry_cnt=retry_cnt + 1)
            semaphore.release()


def filter_shard(vid_shard, url_shard, timestamp_shard, meta_shard, downloaded_ids):
    new_vid_shard = []
    new_url_shard = []
    new_timestamp_shard = []
    new_meta_shard = []
    for i in range(len(vid_shard)):
        if vid_shard[i] not in downloaded_ids:
            new_vid_shard.append(vid_shard[i])
            new_url_shard.append(url_shard[i])
            new_timestamp_shard.append(timestamp_shard[i])
            new_meta_shard.append(meta_shard[i])
    vid_shard = new_vid_shard
    url_shard = new_url_shard
    timestamp_shard = new_timestamp_shard
    meta_shard = new_meta_shard
    return vid_shard, url_shard, timestamp_shard, meta_shard


def download_shard(
    media,
    # input shards
    url_shard,
    id_shard,
    timestamp_shard,
    meta_shard,
    rank,
    process_args: ImageProcessArgs,
    upload_args: UploadArgs,
    # write arguments
    writer,
    output_dir,
    shard_id,
    # downloader arguments
    num_threads=16,
    max_retries=3,
    semaphore_limit=32,
    # message
    message_queue=None,
    # debug
    profile=False,
    debug=False,
):
    os.makedirs(osp.join(output_dir, ".meta"), exist_ok=True)
    error_log = osp.join(output_dir, ".meta", f"{shard_id}.error")

    total = len(url_shard)
    message_queue.put(("START", shard_id, total))

    if timestamp_shard is None:
        timestamp_shard = [None] * total

    # vid_writer = BufferedTextWriter(downloaded_ids, buffer_size=10)
    error_writer = BufferedTextWriter(error_log, buffer_size=1)
    byte_writer = get_writer(
        writer=writer,
        output_dir=output_dir,
        shard_id=shard_id,
        media=media,
        process_args=process_args,
    )

    processor = get_processor(process_args, media=media)

    id_shard, url_shard, timestamp_shard, meta_shard = filter_shard(
        id_shard, url_shard, timestamp_shard, meta_shard, byte_writer.downloaded_ids
    )
    success = len(byte_writer.downloaded_ids)
    message_queue.put(("PROGRESS", shard_id, success))

    download_gen = download_generator(
        url_shard=url_shard,
        id_shard=id_shard,
        timestamp_shard=timestamp_shard,
        meta_shard=meta_shard,
        media=media,
        num_threads=num_threads,
        semaphore_limit=semaphore_limit,
        max_retries=max_retries,
    )

    def process(_id, byte_stream, errorcode):
        # or message_queue won't work
        nonlocal message_queue, success

        if errorcode != 0:
            return

        try:
            if process_args.skip_process:
                byte_writer.write(key=_id, array=byte_stream, fmt="mp4")
            else:
                try:
                    byte_stream = processor(byte_stream)
                except Exception as e:
                    raise ValueError(f"Error while processing {_id}: {e}")
                try:
                    byte_writer.write(key=_id, array=byte_stream, fmt="mp4")
                except Exception as e:
                    raise ValueError(f"Error while writing {_id}: {e}")

            success += 1
            message_queue.put(("PROGRESS", shard_id, 1))

            # vid_writer.write(vid)
        except Exception as e:
            # TODO: something wrong here
            # TypeError("'<' not supported between instances of 'NoneType' and 'NoneType'")
            error_writer.write("\t".join([_id, str(e)]))
            return

    # =================== DEBUG ======================
    if debug:
        print(f"shard_id: {shard_id}")
        for _id, byte_stream, errorcode in tqdm(download_gen):
            # video_bytes, errorcode, msg = download_single(
            #     url=url_shard[i],
            #     timestamp=timestamp_shard[i],
            #     size=process_args.size,
            #     semaphore=Semaphore(1),
            #     media=media,
            # )
            # _id = id_shard[i]
            process(_id, byte_stream, errorcode)
    # ================================================

    for _id, byte_stream, errorcode in download_gen:
        try:
            process(_id, byte_stream, errorcode)
        except Exception as e:
            print(f"Error while processing {_id}: {e}")

    byte_writer.close()

    message_queue.put(("END", shard_id))

    return shard_id, success, total


class ApipeWait:
    def __init__(self):
        self.st = time.time()

    def __call__(self, not_done, interval=5.0):
        done = set(future for future in not_done if future.ready())
        not_done = not_done - done
        while time.time() - self.st < interval:
            pass

        self.st = time.time()

        return done, not_done


apipe_wait = ApipeWait()


def process_message_queue(message_queue, progress, mapping):
    while not message_queue.empty():
        message = message_queue.get()
        if message[0] == "START":
            shard_id, total = message[1:]
            task_id = progress.add_task(f"Shard {shard_id}", total=total)
            mapping[shard_id] = task_id
        elif message[0] == "PROGRESS":
            shard_id, success = message[1:]
            task_id = mapping[shard_id]
            progress.update(task_id, advance=success)
        elif message[0] == "END":
            shard_id = message[1]
            task_id = mapping[shard_id]
            progress.remove_task(task_id)
            mapping.pop(shard_id)

    progress.refresh()


def maybe_upload(output_dir, tar_filenames, upload_args):
    tar_paths = [osp.join(output_dir, _) for _ in tar_filenames]
    for tar_path in tar_paths:
        assert osp.exists(tar_path), f"{tar_path} not found"

    logger.info(f"Found {len(tar_paths)} files to upload")

    if upload_args.upload_hf:
        while True:
            # failproof to multiple uploads
            try:
                hf_api = HfApi()
                hf_api.upload_folder(
                    repo_id=upload_args.repo_id,
                    folder_path=output_dir,
                    allow_patterns=tar_filenames,
                    revision="main",
                    repo_type="dataset",
                )
                break
            except Exception as e:
                logger.error(e)
                time.sleep(10)

        # cmd = (
        #     f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload --repo-type dataset --private "
        #     f"{upload_args.repo_id} {output_dir} --include {' '.join(tar_filenames)}"
        # )
        # run_cmd(cmd=cmd, verbose=True)

    if upload_args.upload_s3:
        ret = run_cmd(
            f"aws s3 cp {' '.join(tar_paths)} s3://sg-sail-home-wangjing/home/{upload_args.bucket} --endpoint-url {upload_args.endpoint_url}",
            async_cmd=False,
        )
        returncode = ret.returncode
        assert returncode == 0, f"Failed to upload, error code: {ret.stderr}"
        logger.info(f"Uploaded to {upload_args.bucket} via S3")

    if upload_args.delete_local:
        run_cmd(f"rm -rf {' '.join(tar_paths)}", async_cmd=False)
        logger.info(f"Deleted local files")


def download(
    media: str,
    sharder: Sharder,
    output_dir,
    process_args: ImageProcessArgs,
    upload_args: UploadArgs,
    writer="file",
    num_processes=16,
    rank=0,
    # downloader arguments
    num_threads=32,
    max_retries=3,
    semaphore_limit=128,
    # debug
    profile=False,
    debug=False,
):
    manager = multiprocessing.Manager()
    message_queue = manager.Queue()

    def launch_job(local_shard_id):

        url_shard, id_shard, timestamp_shard, meta_shard = sharder.fetch_shard(local_shard_id)
        global_shard_id = sharder.get_global_id(local_shard_id)
        _, success, total = download_shard(
            media=media,
            url_shard=url_shard,
            id_shard=id_shard,
            timestamp_shard=timestamp_shard,
            meta_shard=meta_shard,
            rank=rank,
            output_dir=output_dir,
            process_args=process_args,
            upload_args=upload_args,
            writer=writer,
            shard_id=global_shard_id,
            max_retries=max_retries,
            semaphore_limit=semaphore_limit,
            num_threads=num_threads,
            message_queue=message_queue,
            profile=profile,
            debug=debug,
        )
        return global_shard_id, success, total

    num_shards = len(sharder)
    finished_shard_meta = osp.join(output_dir, ".meta", "finished_shards.txt")
    global_shard_ids = [sharder.get_global_id(i) for i in range(0, num_shards)]  # global shard ids processed in current rank

    if not osp.exists(finished_shard_meta):
        with open(finished_shard_meta, "w") as f:
            pass

    with open(finished_shard_meta, "r") as f:
        lines = [_.strip() for _ in f.readlines() if len(_.strip()) > 0]
        finished_shard_ids = [int(_.split("\t")[0]) for _ in lines]
        finished_shard_ids = [_ for _ in finished_shard_ids if _ in global_shard_ids]

    executor = mp.ProcessPool(num_processes)

    global_shard_ids = [i for i in global_shard_ids if i not in finished_shard_ids]
    local_shard_ids = [sharder.get_local_id(i) for i in global_shard_ids]

    if len(global_shard_ids) == 0:
        logger.info(f"All shards have been downloaded in rank {rank}")
        return

    # ======================== DEBUG ========================
    if debug:
        for local_shard_id in local_shard_ids:
            launch_job(local_shard_id)
    # ======================================================

    not_done = set()
    for shard_id in local_shard_ids:
        tar_path = osp.join(output_dir, shardid2name(shard_id) + ".tar")
        cache_dir = osp.join(output_dir, "cache." + shardid2name(shard_id))
        downloaded_but_not_uploaded = osp.exists(tar_path) and not osp.exists(cache_dir)
        if downloaded_but_not_uploaded:
            logger.info(f"Shard {shard_id} already downloaded (maybe not uploaded)")
            continue

        future = executor.apipe(launch_job, local_shard_id=shard_id)
        not_done.add(future)

    progress = get_rich_progress_mofn(disable=False)
    progress.start()
    task_mapping = {}

    finished_rows = []

    while len(not_done) > 0:
        done, not_done = apipe_wait(not_done, interval=3.0)

        process_message_queue(
            message_queue=message_queue,
            progress=progress,
            mapping=task_mapping,
        )

        for future in done:
            global_shard_id, success, total = future.get()
            finished_rows += ["\t".join([str(global_shard_id), str(success), str(total)])]

    progress.stop()

    # ======================== Upload ========================
    tar_filenames = [shardid2name(_) + ".tar" for _ in global_shard_ids]
    maybe_upload(output_dir, tar_filenames, upload_args)

    # ======================== Success ========================
    # Here finished means downloaded, processed and (maybe) uploaded
    with safe_open(finished_shard_meta, "a") as f:
        f.write("\n".join(finished_rows) + "\n")

    df = pd.read_csv(finished_shard_meta, names=["shard_id", "success", "total"], delimiter="\t")
    total_downloaded = df["total"].sum()
    total_finished = df["success"].sum()

    logger.info(
        "\n".join(
            [
                "",
                f"Total downloaded {total_downloaded}",
                f"Total finished {total_finished}",
                f"{sharder.row_count} videos/images",
            ]
        )
    )
