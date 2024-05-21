from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ProcessPoolExecutor
from threading import Semaphore
import multiprocessing as mp
from loguru import logger
import tempfile
from dataclasses import dataclass
from typing import List
import os, os.path as osp
import multiprocessing as mp

from .writer import BufferedTextWriter, get_writer

from kn_util.data.video import download_youtube_as_bytes, read_frames_decord, array_to_video_bytes
from kn_util.data.video.download import StorageLogger
from kn_util.utils.rich import get_rich_progress_mofn
from kn_util.utils.io import load_csv
from kn_util.utils.download import MultiThreadDownloaderInMem
from kn_util.utils.error import SuppressStdoutStderr


@dataclass
class ProcessArguments:
    fps: int = 8
    size: int = 512
    max_size: int = 512
    center_crop: bool = False


def _download_single_youtube(url, semaphore, size):
    semaphore.acquire()
    # use a fake logger to suppress output and capture error
    storage_logger = StorageLogger()

    # default video format from video2dataset
    # https://github.com/iejMac/video2dataset/blob/main/video2dataset/data_reader.py
    video_format = f"wv*[height>={size}][ext=mp4]/" f"w[height>={size}][ext=mp4]/" "bv/b[ext=mp4]"
    video_bytes, errorcode = download_youtube_as_bytes(url, video_format=video_format, logger=storage_logger)

    semaphore.release()

    return video_bytes, errorcode, storage_logger.storage["error"]


def _download_single_direct(url, semaphore):
    semaphore.acquire()
    try:
        downloader = MultiThreadDownloaderInMem(verbose=False, max_retries=5, num_threads=4)
        video_bytes = downloader.download(url)
    except Exception as e:
        semaphore.release()
        return None, 1, str(e)

    semaphore.release()

    return video_bytes, 0, None


def download_single(url, semaphore, size):
    video_bytes = None
    errorcode = 0
    error_str = None

    if "youtube.com" in url:
        video_bytes, errorcode, error_str = _download_single_youtube(url, semaphore=semaphore, size=size)
    elif url.endswith(".mp4") or url.endswith(".avi") or url.endswith(".mkv"):
        video_bytes, errorcode, error_str = _download_single_direct(url, semaphore=semaphore)

    return video_bytes, errorcode, error_str


def download_generator(
    url_shard,
    vid_shard,
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

    def submit_job(url, vid, meta, retry_cnt=0):
        future = executor.submit(
            download_single,
            url=url,
            size=size,
            semaphore=semaphore,
        )
        future._input = dict(meta=meta, vid=vid, url=url, retry_cnt=retry_cnt)
        return future

    def submit_jobs(ids):
        futures = set()
        for i in ids:
            future = submit_job(url_shard[i], vid_shard[i], meta_shard[i], retry_cnt=0)
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
            vid = future._input["vid"]
            retry_cnt = future._input["retry_cnt"]

            if errorcode == 0:
                yield vid, video_bytes, errorcode
            else:
                yield vid, None, errorcode
                if retry_cnt + 1 < max_retries:
                    submit_job(url=url, vid=vid, meta=meta, retry_cnt=retry_cnt + 1)


def filter_shard(vid_shard, url_shard, meta_shard, downloaded_meta):
    if osp.exists(downloaded_meta):
        with open(downloaded_meta, "r") as f:
            downloaded_vid = f.readlines()
        new_vid_shard = []
        new_url_shard = []
        new_meta_shard = []
        for i in range(len(vid_shard)):
            if vid_shard[i] not in downloaded_vid:
                new_vid_shard.append(vid_shard[i])
                new_url_shard.append(url_shard[i])
                new_meta_shard.append(meta_shard[i])
        vid_shard = new_vid_shard
        url_shard = new_url_shard
        meta_shard = new_meta_shard
    return vid_shard, url_shard, meta_shard


def download_shard(
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
):
    os.makedirs(osp.join(output_dir, ".meta"), exist_ok=True)
    downloaded_vids = osp.join(output_dir, ".meta", f"downloaded_vids.r{rank}.csv")
    error_log = osp.join(output_dir, ".meta", f"{shard_id}.error")

    total = len(url_shard)
    failed = 0

    if osp.exists(downloaded_vids):
        logger.info(f"Found downloaded videos, filtering shard {shard_id}")
        vid_shard, url_shard, meta_shard = filter_shard(vid_shard, url_shard, meta_shard, downloaded_vids)
        logger.info(f"Filtered shard {shard_id} to {len(vid_shard)} videos")

    vid_writer = BufferedTextWriter(downloaded_vids, buffer_size=10)
    error_writer = BufferedTextWriter(error_log, buffer_size=1)
    byte_writer = get_writer(writer=writer, output_dir=output_dir, shard_id=shard_id)

    download_gen = download_generator(
        url_shard=url_shard,
        vid_shard=vid_shard,
        meta_shard=meta_shard,
        num_threads=num_threads,
        semaphore_limit=semaphore_limit,
        max_retries=max_retries,
    )

    for vid, video_bytes, errorcode in download_gen:
        # process
        if errorcode != 0:
            continue

        try:
            with SuppressStdoutStderr():
                frames = read_frames_decord(
                    video_bytes,
                    fps=process_args.fps,
                    size=process_args.size,
                    max_size=process_args.max_size,
                    is_bytes=True,
                    output_format="thwc",
                )

                video_bytes = array_to_video_bytes(frames, fps=process_args.fps)

                # write video
                byte_writer.write(key=vid, video_bytes=video_bytes)

            vid_writer.write(vid)
        except Exception as e:
            error_writer.write("\t".join([vid, str(e)]))
            failed += 1
            continue

    return shard_id, failed, total


def download(
    sharder,
    output_dir,
    process_args: ProcessArguments,
    writer="file",
    num_processes=16,
    rank=0,
    # downloader arguments
    num_threads=32,
    max_retries=3,
    semaphore_limit=128,
):
    num_shards = len(sharder)

    executor = ProcessPoolExecutor(num_processes)
    shard_cnt = 0

    shard_cnt += num_processes

    # ======================== DBEUG ========================
    # url_shard, vid_shard, meta_shard = sharder.fetch_shard(0)
    # download_shard(
    #     url_shard=url_shard,
    #     vid_shard=vid_shard,
    #     meta_shard=meta_shard,
    #     output_dir=output_dir,
    #     writer=writer,
    #     shard_id=0,
    #     process_args=process_args,
    #     max_retries=max_retries,
    #     semaphore_limit=semaphore_limit,
    #     num_threads=num_threads,
    # )
    # import ipdb

    # ipdb.set_trace()
    # ======================================================

    def submit_jobs(shard_ids):
        shards, shard_ids = sharder.fetch_shards(shard_ids)
        futures = set()
        for shard_id, (url_shard, vid_shard, meta_shard) in zip(shard_ids, shards):
            future = executor.submit(
                download_shard,
                url_shard=url_shard,
                vid_shard=vid_shard,
                meta_shard=meta_shard,
                rank=rank,
                output_dir=output_dir,
                process_args=process_args,
                writer=writer,
                shard_id=shard_id,
                max_retries=max_retries,
                semaphore_limit=semaphore_limit,
                num_threads=num_threads,
            )
            futures.add(future)
        return futures

    init_shard_ids = list(range(min(num_processes, num_shards)))
    not_done = submit_jobs(init_shard_ids)

    total_finished = 0
    total_downloaded = 0
    history_finished = 0

    # polling for latest progress
    while len(not_done) != 0:
        ongoing_finished = 0

        done, not_done = wait(not_done, return_when=FIRST_COMPLETED, timeout=5.0)

        # deal with finished shards
        for future in done:
            shard_id, failed, total = future.result()
            logger.info(f"Shard {shard_id} downloaded {total - failed}/{total} videos")
            history_finished += total
            total_downloaded += total - failed

        total_finished = history_finished + ongoing_finished

        num_shard_to_submit = min(len(done), num_shards - shard_cnt)
        shard_ids = list(range(shard_cnt, shard_cnt + num_shard_to_submit))
        shard_cnt += num_shard_to_submit
        futures = submit_jobs(shard_ids)

        not_done = not_done.union(futures)

    logger.info(
        "\n".join(
            [
                f"Total downloaded {total_downloaded}",
                f"Total finished {total_finished}",
                f"{sharder.row_count} videos",
            ]
        )
    )

    executor.shutdown(wait=True)
