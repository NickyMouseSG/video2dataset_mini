from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ProcessPoolExecutor
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

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from .writer import BufferedTextWriter, get_writer
from .sharder import Sharder
from .utils import shardid2name, safe_open

from kn_util.data.video import download_youtube_as_bytes, read_frames_decord
from kn_util.data.video.download import StorageLogger
from kn_util.utils.rich import get_rich_progress_mofn
from kn_util.utils.download import MultiThreadDownloaderInMem
from kn_util.utils.error import SuppressStdoutStderr
from kn_util.utils.system import run_cmd
from kn_util.tools.lfs import upload_files


@dataclass
class ProcessArguments:
    disabled: bool = False
    fps: int = 8
    size: int = 512
    max_size: int = 512
    center_crop: bool = False
    quality: int = 9


def _download_single_youtube(url, size):
    # use a fake logger to suppress output and capture error
    storage_logger = StorageLogger()

    # default video format from video2dataset
    # https://github.com/iejMac/video2dataset/blob/main/video2dataset/data_reader.py
    video_format = f"wv*[height>={size}][ext=mp4]/" f"w[height>={size}][ext=mp4]/" "bv/b[ext=mp4]"
    video_bytes, errorcode = download_youtube_as_bytes(url, video_format=video_format, logger=storage_logger)

    return video_bytes, errorcode, storage_logger.storage["error"]


def _download_single_direct(url):
    try:
        downloader = MultiThreadDownloaderInMem(verbose=False, max_retries=0, num_threads=1)
        video_bytes = downloader.download(url)
    except Exception as e:
        return None, 1, str(e)

    return video_bytes, 0, None


def download_single(url, size, semaphore, media="video"):
    _bytes = None
    errorcode = 0
    error_str = None
    semaphore.acquire()

    if media == "video":
        if "youtube.com" in url:
            _bytes, errorcode, error_str = _download_single_youtube(url, size=size)
        elif url.endswith(".mp4") or url.endswith(".avi") or url.endswith(".mkv"):
            _bytes, errorcode, error_str = _download_single_direct(url)
    elif media == "image":
        _bytes, errorcode, error_str = _download_single_direct(url)
    else:
        raise ValueError(f"Invalid media type: {media}")

    return _bytes, errorcode, error_str


def download_generator(
    url_shard,
    id_shard,
    meta_shard,
    # download args
    size=360,
    # parallel
    num_threads=16,
    semaphore_limit=32,
    max_retries=3,
):
    executor = ThreadPoolExecutor(num_threads)
    semaphore = Semaphore(semaphore_limit)

    def submit_job(url, _id, meta, retry_cnt=0):
        future = executor.submit(
            download_single,
            url=url,
            size=size,
            semaphore=semaphore,
        )
        future._input = dict(meta=meta, id=_id, url=url, retry_cnt=retry_cnt)
        return future

    def submit_jobs(ids):
        futures = set()
        for i in ids:
            future = submit_job(url_shard[i], id_shard[i], meta_shard[i], retry_cnt=0)
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
            retry_cnt = future._input["retry_cnt"]

            if errorcode == 0:
                yield _id, video_bytes, errorcode
            else:
                yield _id, None, errorcode
                if retry_cnt + 1 < max_retries:
                    submit_job(url, _id, meta, retry_cnt=retry_cnt + 1)
            semaphore.release()


def filter_shard(vid_shard, url_shard, meta_shard, downloaded_ids):
    new_vid_shard = []
    new_url_shard = []
    new_meta_shard = []
    for i in range(len(vid_shard)):
        if vid_shard[i] not in downloaded_ids:
            new_vid_shard.append(vid_shard[i])
            new_url_shard.append(url_shard[i])
            new_meta_shard.append(meta_shard[i])
    vid_shard = new_vid_shard
    url_shard = new_url_shard
    meta_shard = new_meta_shard
    return vid_shard, url_shard, meta_shard


def download_shard(
    media,
    # input shards
    url_shard,
    vid_shard,
    meta_shard,
    # rank
    rank,
    # process arguments
    process_args: ProcessArguments,
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
):
    os.makedirs(osp.join(output_dir, ".meta"), exist_ok=True)
    error_log = osp.join(output_dir, ".meta", f"{shard_id}.error")

    total = len(url_shard)
    message_queue.put(("START", shard_id, total))

    # vid_writer = BufferedTextWriter(downloaded_ids, buffer_size=10)
    error_writer = BufferedTextWriter(error_log, buffer_size=1)
    byte_writer = get_writer(writer=writer, output_dir=output_dir, shard_id=shard_id, media=media, process_args=process_args)

    id_shard, url_shard, meta_shard = filter_shard(vid_shard, url_shard, meta_shard, byte_writer.downloaded_ids)
    success = len(byte_writer.downloaded_ids)
    message_queue.put(("PROGRESS", shard_id, success))

    download_gen = download_generator(
        url_shard=url_shard,
        id_shard=id_shard,
        meta_shard=meta_shard,
        num_threads=num_threads,
        semaphore_limit=semaphore_limit,
        max_retries=max_retries,
    )

    st = time.time()

    for _id, _bytes, errorcode in download_gen:
        if profile:
            logger.info(f"Downloaded {_id} in {time.time() - st:.2f}s")
            st = time.time()

        if errorcode != 0:
            continue

        # ======================== Process Video ========================
        if media == "video":
            try:
                if process_args.disabled:
                    byte_writer.write(key=_id, array=_bytes, fmt="mp4")
                else:
                    frames, video_meta = read_frames_decord(
                        _bytes,
                        fps=process_args.fps,
                        size=process_args.size,
                        max_size=process_args.max_size,
                        is_bytes=True,
                        output_format="thwc",
                        return_meta=True,
                    )

                    if profile:
                        logger.info(f"Processed {_id} in {time.time() - st:.2f}s")
                        st = time.time()

                    # write video
                    byte_writer.write(key=_id, array=frames, video_meta=video_meta)

                success += 1
                message_queue.put(("PROGRESS", shard_id, 1))

                if profile:
                    logger.info(f"Written {_id} in {time.time() - st:.2f}s")

                # vid_writer.write(vid)
            except Exception as e:
                error_writer.write("\t".join([_id, str(e)]))
                continue

        elif media == "image":
            transforms = []
            transforms.append(IT.Resize(size=process_args.size, max_size=process_args.max_size))
            if process_args.center_crop:
                transforms.append(IT.CenterCrop(size=process_args.size))
            transforms = IT.Compose(transforms)

            try:
                array = Image.open(io.BytesIO(_bytes)).convert("RGB")
                array = torch.from_numpy(np.array(array))
                array = transforms(array)
                array = array.numpy()

                # write video
                byte_writer.write(key=_id, array=frames, fmt="jpeg")

                success += 1

            except Exception as e:
                error_writer.write("\t".join([_id, str(e)]))
                continue

        st = time.time()

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


def download(
    media: str,
    sharder: Sharder,
    output_dir,
    process_args: ProcessArguments,
    writer="file",
    num_processes=16,
    rank=0,
    # downloader arguments
    num_threads=32,
    max_retries=3,
    semaphore_limit=128,
    # upload,
    upload=False,
    repo_id=None,
    # debug
    profile=False,
):
    manager = multiprocessing.Manager()
    message_queue = manager.Queue()

    def launch_job(local_shard_id):
        url_shard, vid_shard, meta_shard = sharder.fetch_shard(local_shard_id)
        global_shard_id = sharder.get_global_id(local_shard_id)
        _, success, total = download_shard(
            media=media,
            url_shard=url_shard,
            vid_shard=vid_shard,
            meta_shard=meta_shard,
            rank=rank,
            output_dir=output_dir,
            process_args=process_args,
            writer=writer,
            shard_id=global_shard_id,
            max_retries=max_retries,
            semaphore_limit=semaphore_limit,
            num_threads=num_threads,
            message_queue=message_queue,
            profile=profile,
        )
        return global_shard_id, success, total

    num_shards = len(sharder)
    downloaded_shard_meta = osp.join(output_dir, ".meta", "downloaded_shard.txt")
    global_shard_ids = [sharder.get_global_id(i) for i in range(0, num_shards)]  # global shard ids processed in current rank

    if not osp.exists(downloaded_shard_meta):
        with open(downloaded_shard_meta, "w") as f:
            pass

    with open(downloaded_shard_meta, "r") as f:
        lines = [_.strip() for _ in f.readlines() if len(_.strip()) > 0]
        downloaded_shard_ids = [int(_.split("\t")[0]) for _ in lines]
        downloaded_shard_ids = [_ for _ in downloaded_shard_ids if _ in global_shard_ids]

    executor = mp.ProcessPool(num_processes)

    local_shard_ids = [sharder.get_local_id(i) for i in global_shard_ids if i not in downloaded_shard_ids]

    # ======================== DBEUG ========================
    # launch_job(local_shard_ids[0])
    # import ipdb

    # ipdb.set_trace()
    # ======================================================

    not_done = set()
    for shard_id in local_shard_ids:
        future = executor.apipe(launch_job, local_shard_id=shard_id)
        not_done.add(future)

    progress = get_rich_progress_mofn()
    progress.start()
    task_mapping = {}

    while len(not_done) > 0:
        done, not_done = apipe_wait(not_done, interval=10.0)

        process_message_queue(
            message_queue=message_queue,
            progress=progress,
            mapping=task_mapping,
        )

        for future in done:
            global_shard_id, success, total = future.get()
            with safe_open(downloaded_shard_meta, "a") as f:
                f.write("\t".join([str(global_shard_id), str(success), str(total)]) + "\n")

    progress.stop()

    df = pd.read_csv(downloaded_shard_meta, names=["shard_id", "success", "total"], delimiter="\t")
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

    if upload:
        while True:
            if osp.isdir("~repo"):
                # this means some other ranks is uploading
                time.sleep(10)
            else:
                import ipdb; ipdb.set_trace()
                os.system(f"GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/k-nick/{repo_id}.git ~repo")
                shard_names = [osp.join(output_dir, shardid2name(_)) + ".tar" for _ in global_shard_ids]
                os.system("mv " + " ".join(shard_names) + " ~repo")
                os.chdir("~repo")
                upload_files(batch_size=30)
                os.system(f"rm -rf ~repo")
                break
