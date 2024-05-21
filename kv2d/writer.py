import webdataset as wds
import os, os.path as osp
import numpy as np
import multiprocessing as mp
import tempfile

from kn_util.utils.system import run_cmd
from kn_util.data.video import save_video_imageio, array_to_video_bytes


class TarWriter:
    def __init__(self, tar_file):
        fd = open(tar_file, "wb")
        self.wds_writer = wds.TarWriter(fd)

    def write(self, key, frames, fmt="mp4", video_meta=None):
        video_bytes = array_to_video_bytes(frames, fps=video_meta["fps"])
        self.wds_writer.write({"__key__": key, fmt: video_bytes})

    @property
    def downloaded_vids(self):
        return set()  # cannot resume vids from tar file

    def close(self):
        self.wds_writer.close()


class FileWriter:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def write(self, key, frames, fmt="mp4", video_meta=None):
        cache_file = osp.join(self.cache_dir, f"{key}.{fmt}")
        with tempfile.NamedTemporaryFile(suffix="." + fmt, delete=False) as f:
            save_video_imageio(frames, f.name, fps=video_meta["fps"])
            run_cmd(f"mv {f.name} {cache_file}")

    @property
    def downloaded_vids(self):
        cache_files = set(os.listdir(self.cache_dir))
        downloaded_vids = set()
        for cache_file in cache_files:
            key, _ = cache_file.split(".")
            downloaded_vids.add(key)

        return downloaded_vids

    def close(self):
        pass


class CachedTarWriter(FileWriter):
    def __init__(self, tar_file, cache_dir):
        super().__init__(cache_dir)
        self.tar_file = tar_file

    def close(self):
        writer = wds.TarWriter(open(self.tar_file, "wb"))
        for cache_file in os.listdir(self.cache_dir):
            with open(osp.join(self.cache_dir, cache_file), "rb") as f:
                video_bytes = f.read()
                key, fmt = cache_file.split(".")
                writer.write({"__key__": key, fmt: video_bytes})

        run_cmd(f"rm -rf {self.cache_dir}", async_cmd=True)


def get_writer(writer, output_dir, shard_id):
    cache_dir = osp.join(output_dir, f"cache.{shard_id:04d}")
    tar_file = osp.join(output_dir, f"{shard_id:04d}.tar")
    if writer == "tar":
        return TarWriter(tar_file)
    elif writer == "file":
        return FileWriter(output_dir)
    elif writer == "cached_tar":
        return CachedTarWriter(tar_file, cache_dir=cache_dir)
    else:
        raise ValueError(f"Invalid writer: {writer}")


class BufferedTextWriter:
    def __init__(self, output_file, buffer_size=100):
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.buffer = []

        if not osp.exists(self.output_file):
            self.handler = open(self.output_file, "w")
            self._records = []
        else:
            self.handler = open(self.output_file, "a+")
            self._records = self.handler.readlines()

        self.lock = mp.Lock()

    @property
    def records(self):
        return self._records

    def write(self, cur_str):
        self.buffer.append(cur_str)

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        with self.lock:
            self.handler.write("\n".join(self.buffer) + "\n")
        self.buffer = []

    def close(self):
        self.flush()
        self.handler.close()
