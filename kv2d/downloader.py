from kn_util.data.video import YTDLPDownloader
from kn_util.utils.rich import get_rich_progress_mofn
from kn_util.utils.io import load_csv
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ProcessPoolExecutor
from threading import Semaphore
import multiprocessing as mp
import os.path as osp
from kn_util.utils.logger import StorageLogger

from loguru import logger


def download_single(url, meta, path, semaphore, retry_cnt):
    semaphore.acquire()
    # use a fake logger to suppress output and capture error
    storage_logger = StorageLogger()

    # default video format from video2dataset
    # https://github.com/iejMac/video2dataset/blob/main/video2dataset/data_reader.py
    video_size = 360
    video_format = f"wv*[height>={video_size}][ext=mp4]/" f"w[height>={video_size}][ext=mp4]/" "bv/b[ext=mp4]"

    errorcode = YTDLPDownloader.download(
        youtube_id=url,
        video_path=path,
        video_format=video_format,
        quiet=True,
        logger=storage_logger,
    )
    semaphore.release()

    return url, meta, path, errorcode, retry_cnt, storage_logger.storage["error"]


def download_shard(
    url_shard,
    vid_shard,
    meta_shard,
    output_dir,
    shard_id,
    progress_dict,
    num_threads=16,
    max_retries=3,
    semaphore_limit=32,
):
    executor = ThreadPoolExecutor(num_threads)
    semaphore = Semaphore(semaphore_limit)
    failed = 0
    total = len(url_shard)
    download_meta = osp.join(output_dir, ".meta", f".downloaded.shard{shard_id:02d}.tsv")
    if not osp.exists(download_meta):
        download_meta_fp = open(download_meta, "w")
        downloaded_vid = []
    else:
        download_meta_fp = open(download_meta, "r+")
        download_meta_list = load_csv(download_meta, delimiter="\t", has_header=False)
        downloaded_vid = [_[1] for _ in download_meta_list]

    if len(downloaded_vid) > 0:
        logger.info(f"Shard {shard_id} resume from {len(downloaded_vid)} videos")

    not_done = []
    num_downloaded = 0

    for url, vid, meta in zip(url_shard, vid_shard, meta_shard):
        if vid in downloaded_vid:
            num_downloaded += 1
            continue
        not_done.append(
            executor.submit(
                download_single,
                url=url,
                meta=meta,
                path=osp.join(output_dir, vid + ".mp4"),
                retry_cnt=0,
                semaphore=semaphore,
            )
        )

    progress_dict[shard_id] = num_downloaded

    while len(not_done) > 0:
        done, not_done = wait(not_done, return_when=FIRST_COMPLETED, timeout=30.0)

        success_metas = []
        failed_metas = []
        finish_cnt = 0
        for future in done:
            url, meta, path, errorcode, retry_cnt, error = future.result()
            if errorcode == 0:
                success_metas += [(url, meta, "success")]
                finish_cnt += 1
            else:
                is_unavail = "Video unavailable" in error[0]
                is_private = "Private video" in error[0]
                is_illegal = "violating YouTube's" in error[0]

                is_common_error = is_unavail or is_private or is_illegal

                if not is_common_error and retry_cnt + 1 < max_retries:
                    logger.info(error[0], f"Retry {retry_cnt + 1}/{max_retries}")
                    executor.submit(
                        download_single,
                        url=url,
                        meta=meta,
                        path=path,
                        retry_cnt=retry_cnt + 1,
                        semaphore=semaphore,
                    )
                else:
                    failed_metas += [(url, meta, error[0].strip())]
                    failed += 1
                    finish_cnt += 1

        progress_dict[shard_id] += finish_cnt

        if len(success_metas) + len(failed_metas) > 0:
            meta_str = "\n".join([f"{_[0]}\t{_[1]}\t{_[2]}" for _ in success_metas] + [f"{_[0]}\t{_[1]}\t{_[2]}" for _ in failed_metas])
            download_meta_fp.write(meta_str + "\n")
            download_meta_fp.flush()

    download_meta_fp.close()
    executor.shutdown(wait=True)

    return shard_id, failed, total


class VideoDownloader:
    def __init__(
        self,
        num_processes=16,
        verbose=False,
        num_threads=32,
        max_retries=3,
        semaphore_limit=128,
    ):
        self.num_processes = num_processes
        self.verbose = verbose
        self.num_threads = num_threads
        self.max_retries = max_retries
        self.semaphore_limit = semaphore_limit

        self.manager = mp.Manager()
        self._progress = self.manager.dict()

    def __exit__(self):
        self.manager.shutdown()

    def download(
        self,
        sharder,
        output_dir,
    ):
        num_processes = self.num_processes
        verbose = self.verbose
        num_shards = len(sharder)

        progress = get_rich_progress_mofn(
            disable=not verbose,
            refresh_per_second=1,
        )
        progress.start()
        progress.add_task("Total", total=sharder.row_count)

        executor = ProcessPoolExecutor(num_processes)
        shard_cnt = 0

        not_done = []
        shards = sharder.fetch_shards(list(range(num_processes)))
        shard_cnt += num_processes

        for shard_id in range(num_processes):
            (
                url_shard,
                vid_shard,
                meta_shard,
            ) = shards[shard_id]
            progress.add_task(f"Shard {shard_id}", total=len(url_shard))
            self._progress[shard_id] = 0

            # https://stackoverflow.com/questions/17419879/why-i-cannot-use-python-module-concurrent-futures-in-class-method
            # here self.download_shard is not working as expected
            future = executor.submit(
                download_shard,
                url_shard=url_shard,
                vid_shard=vid_shard,
                meta_shard=meta_shard,
                output_dir=output_dir,
                shard_id=shard_id,
                progress_dict=self._progress,
            )
            not_done.append(future)

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
                progress.remove_task(shard_id + 1)
                self._progress.pop(shard_id)
                history_finished += total
                total_downloaded += total - failed

            # check ongoing shards
            for shard_id in self._progress.keys():
                cur_progress = self._progress[shard_id]
                ongoing_finished += cur_progress
                progress.update(shard_id + 1, completed=cur_progress)

            total_finished = history_finished + ongoing_finished

            progress.update(0, completed=total_finished)
            progress.refresh()

            num_shard_to_submit = min(len(done), num_shards - shard_cnt)
            shard_ids = list(range(shard_cnt, shard_cnt + num_shard_to_submit))
            shards = sharder.fetch_shards(shard_ids)
            shard_cnt += num_shard_to_submit

            # submit new shards if there are any
            for shard_id, shard in zip(shard_ids, shards):
                url_shard, vid_shard, meta_shard = shard
                progress.add_task(f"Shard {shard_id}", total=len(url_shard))
                self._progress[shard_id] = 0
                future = executor.submit(
                    download_shard,
                    url_shard=url_shard,
                    vid_shard=vid_shard,
                    meta_shard=meta_shard,
                    output_dir=output_dir,
                    shard_id=shard_id,
                    progress_dict=self._progress,
                )
                not_done.add(future)

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

        progress.stop()
