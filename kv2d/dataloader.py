import pyarrow as pa
import pyarrow.csv as pa_csv
import pandas as pd
from threading import Semaphore
from torch.utils.data import IterableDataset, DataLoader
import tempfile
import webdataset as wds
from queue import Queue
import os, os.path as osp

from tqdm import tqdm

from kv2d.sharder import ReadArguments, Sharder
from kv2d.download import download_single, download_generator
from kn_util.data.video import read_frames_decord
from kn_util.utils.download import MultiThreadDownloader, get_hf_headers


class VideoShardIterableDataset(IterableDataset):
    """IterableDataset for reading videos from online shards
    Use DataLoader to prefetch shards online
    Args:
        urls (List[str]): list of urls to download
        num_threads (int): number of threads for downloading
        num_workers (int): number of workers for DataLoader for fetching `shard`
        prefetch_factor (int): prefetch factor for DataLoader
    """

    class _ShardIterableDataset(IterableDataset):
        def __init__(self, urls, num_threads=16, shard_cache_size=4):
            self.urls = urls
            self.downloader = MultiThreadDownloader(
                num_threads=num_threads,
                headers=get_hf_headers(),
                verbose=0,
            )
            self.shard_cache_size = shard_cache_size
            self.shard_cache = Queue()

        def __iter__(self):
            for url in self.urls:
                _last_tempfile = None
                with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as f:
                    self.downloader.download(url, f.name)
                    print(f"Downloaded {url} to {f.name}")
                    yield url, wds.WebDataset(f.name)

    def __init__(self, urls, num_threads=16, num_workers=4, prefetch_factor=2):
        self.urls = urls
        self.shard_gen = DataLoader(
            self._ShardIterableDataset(urls, num_threads=num_threads),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=lambda x: x[0],
        )

    def __iter__(self):
        for url, shard in self.shard_gen:
            for idx, sample in enumerate(shard):
                yield url, idx, sample


class VideoIterableDataset(IterableDataset):
    """IterableDataset for downloading and reading videos on the fly
    Args:
        file_paths (str): path to the file containing the video urls
        read_args (ReadArguments): ReadArguments object
        size (int): size of the frame
        num_frames (int): number of frames to read
        fps (int): frames per second
        num_threads (int): number of threads for downloading
        semaphore_limit (int): semaphore limit for downloading
        rank_id (int): rank id for distributed training
        world_size (int): world size for distributed training
    Returns:
        Tuple[str, np.ndarray, Dict]: id, frames, video_meta
    """

    def __init__(
        self,
        file_paths,
        read_args: ReadArguments,
        num_threads=16,
        size=224,
        semaphore_limit=16,
        rank_id=0,
        world_size=1,
        finished_ids=None,
    ):
        self.read_args = read_args
        self.sharder = Sharder(
            input_file=file_paths,
            read_args=read_args,
            shard_size=10000,
            rank_id=rank_id,
            world_size=world_size,
        )
        self.size = size
        self.semaphore_limit = semaphore_limit
        self.num_threads = num_threads
        self.finished_ids = finished_ids

    def filter_finished_ids(self, url_shard, id_shard, timestamp_shard, meta_shard):
        if self.finished_ids is not None:
            idxs = [idx for idx, _id in enumerate(id_shard) if _id not in self.finished_ids]
            id_shard = [id_shard[idx] for idx in idxs]
            url_shard = [url_shard[idx] for idx in idxs]
            timestamp_shard = [timestamp_shard[idx] for idx in idxs]
            meta_shard = [meta_shard[idx] for idx in idxs]
        return url_shard, id_shard, timestamp_shard, meta_shard

    def __iter__(self):
        for url_shard, id_shard, timestamp_shard, meta_shard in self.sharder:
            timestamp_shard = [None] * len(url_shard) if timestamp_shard is None else timestamp_shard
            url_shard, id_shard, timestamp_shard, meta_shard = self.filter_finished_ids(url_shard, id_shard, timestamp_shard, meta_shard)

            download_gen = download_generator(
                url_shard=url_shard,
                id_shard=id_shard,
                timestamp_shard=timestamp_shard,
                meta_shard=meta_shard,
                size=self.size,
                semaphore_limit=self.semaphore_limit,
                num_threads=self.num_threads,
            )

            for _id, byte_stream, errorcode in download_gen:
                if errorcode == 0:
                    try:
                        # frames, video_meta = read_frames_decord(
                        #     byte_stream,
                        #     num_frames=self.num_frames,
                        #     fps=self.fps,
                        #     is_bytes=True,
                        #     return_meta=True,
                        # )
                        yield _id, byte_stream
                    except:
                        print(f"Error reading {_id}")
                else:
                    print(f"Error downloading {_id}")


if __name__ == "__main__":
    read_args = ReadArguments(headers=True, url_col="contentUrl", id_col="videoid")
    dataloader = VideoIterableDataset(
        file_paths="data/url/mini.tsv",
        read_args=read_args,
        fps=2,
        size=224,
        num_threads=16,
        semaphore_limit=16,
        pin_memory=True,
        num_workers=32,
    )

    for sample in tqdm(dataloader):
        pass
