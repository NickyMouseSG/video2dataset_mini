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

    not_done = []
    num_downloaded = 0
    for url, meta in zip(url_shard, meta_shard):
        if meta["vid"] in downloaded_vid:
            num_downloaded += 1
            continue
        not_done.append(
            executor.submit(
                download_single,
                url=url,
                meta=meta,
                path=osp.join(output_dir, meta["vid"] + ".mp4"),
                retry_cnt=0,
                semaphore=semaphore,
            )
        )

    progress_dict[shard_id] = num_downloaded

    while len(not_done) > 0:
        done, not_done = wait(not_done, return_when=FIRST_COMPLETED, timeout=30.0)

        success_metas = []
        finish_cnt = 0
        for future in done:
            url, meta, path, errorcode, retry_cnt, error = future.result()
            if errorcode == 0:
                success_metas += [(url, meta)]
                finish_cnt += 1
                download_meta_fp.write(f"{url}\t{meta['vid']}\n")
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
                    failed += 1
                    finish_cnt += 1

        progress_dict[shard_id] += finish_cnt

        for url, meta in success_metas:
            download_meta_fp.write(f"{url}\t{meta['vid']}\n")

        download_meta_fp.flush()

    download_meta_fp.close()
    executor.shutdown(wait=True)

    return shard_id, failed, total


class VideoDownloader:
    def __init__(
        self,
        num_processes=16,
        verbose=False,
        shard_size=1000,
        num_threads=32,
        max_retries=3,
        semaphore_limit=128,
    ):
        self.num_processes = num_processes
        self.verbose = verbose
        self.num_threads = num_threads
        self.max_retries = max_retries
        self.semaphore_limit = semaphore_limit
        self.shard_size = shard_size

        self.manager = mp.Manager()
        self._progress = self.manager.dict()

    def __exit__(self):
        self.manager.shutdown()

    def download(
        self,
        urls,
        metas,
        output_dir,
    ):
        num_processes = self.num_processes
        verbose = self.verbose

        # shard_size = (len(urls) + num_processes - 1) // num_processes
        shard_size = self.shard_size
        ranges = [(i, min(i + shard_size, len(urls))) for i in range(0, len(urls), shard_size)]
        assert len(ranges) >= num_processes, "Shard size is too small, consider increasing shard size or decreasing num_processes"

        url_shards = [urls[start:end] for start, end in ranges]
        meta_shards = [metas[start:end] for start, end in ranges]

        progress = get_rich_progress_mofn(
            disable=not verbose,
            refresh_per_second=1,
        )
        progress.start()
        progress.add_task("Total", total=len(urls))

        # self.download_shard(url_shards[0], meta_shards[0], output_dir, 0)
        # import ipdb; ipdb.set_trace()

        executor = ProcessPoolExecutor(num_processes)

        gen = iter(enumerate(zip(url_shards, meta_shards)))

        not_done = []
        for _ in range(num_processes):
            shard_id, (url_shard, meta_shard) = next(gen)
            progress.add_task(f"Shard {shard_id}", total=ranges[shard_id][1] - ranges[shard_id][0])
            self._progress[shard_id] = 0
            # https://stackoverflow.com/questions/17419879/why-i-cannot-use-python-module-concurrent-futures-in-class-method
            # here self.download_shard is not working as expected
            future = executor.submit(
                download_shard,
                url_shard=url_shard,
                meta_shard=meta_shard,
                output_dir=output_dir,
                shard_id=shard_id,
                progress_dict=self._progress,
            )
            not_done.append(future)

        finished_downloaded = 0

        # polling for latest progress
        while len(not_done) != 0:
            ongoing_downloaded = 0

            done, not_done = wait(not_done, return_when=FIRST_COMPLETED, timeout=5.0)
            for shard_id in self._progress.keys():
                cur_progress = self._progress[shard_id]
                ongoing_downloaded += cur_progress
                progress.update(shard_id + 1, completed=cur_progress)

            for future in done:
                shard_id, failed, total = future.result()
                logger.info(f"Shard {shard_id} downloaded {total - failed}/{total} videos")
                progress.remove_task(shard_id + 1)
                self._progress.pop(shard_id)
                finished_downloaded += total

            total_progress = finished_downloaded + ongoing_downloaded

            progress.update(0, completed=total_progress)
            progress.refresh()

            for _ in range(len(done)):
                try:
                    shard_id, (url_shard, meta_shard) = next(gen)
                    progress.add_task(f"Shard {shard_id}", total=len(url_shard))
                    self._progress[shard_id] = 0
                    future = executor.submit(
                        download_shard,
                        url_shard=url_shard,
                        meta_shard=meta_shard,
                        output_dir=output_dir,
                        shard_id=shard_id,
                        progress_dict=self._progress,
                    )
                    not_done.add(future)
                except StopIteration:
                    break

        logger.info(f"Total downloaded {total_downloaded}/{len(urls)} videos")

        executor.shutdown(wait=True)

        progress.stop()
