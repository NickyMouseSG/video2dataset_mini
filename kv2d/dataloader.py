import pyarrow as pa
import pyarrow.csv as pa_csv
import pandas as pd
from threading import Semaphore
from torch.utils.data import IterableDataset, DataLoader

from tqdm import tqdm

from kv2d.sharder import ReadArguments, Sharder
from kv2d.download import download_single, download_generator
from kn_util.data.video import read_frames_decord


class VideoIterableDataset(IterableDataset):
    def __init__(
        self,
        file_paths,
        read_args: ReadArguments,
        size=224,
        # decord args
        num_frames=None,
        fps=None,
        num_threads=16,
        semaphore_limit=16,
        rank_id=0,
        world_size=1,
    ):
        self.read_args = read_args
        self.sharder = Sharder(
            input_file=file_paths,
            read_args=read_args,
            shard_size=10000,
            rank_id=rank_id,
            world_size=world_size,
        )
        self.num_frames = num_frames
        self.fps = fps
        self.size = size
        self.semaphore_limit = semaphore_limit
        self.num_threads = num_threads

    def __iter__(self):
        for url_shard, id_shard, timestamp_shard, meta_shard in self.sharder:
            timestamp_shard = [None] * len(url_shard) if timestamp_shard is None else timestamp_shard

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
                    frames, video_meta = read_frames_decord(
                        byte_stream,
                        num_frames=self.num_frames,
                        fps=self.fps,
                        is_bytes=True,
                        return_meta=True,
                    )
                    yield _id, frames, video_meta
                else:
                    print(f"Error downloading {_id}")


class VideoDataLoader(DataLoader):
    def __init__(self, *args, num_workers=0, pin_memory=True, **kwargs):
        dataset = VideoIterableDataset(*args, **kwargs)
        super().__init__(dataset=dataset, num_workers=num_workers, pin_memory=pin_memory, collate_fn=lambda x: x)


if __name__ == "__main__":
    read_args = ReadArguments(headers=True, url_col="contentUrl", id_col="videoid")
    dataloader = VideoDataLoader(
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
