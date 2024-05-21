import webdataset as wds
import os, os.path as osp
import numpy as np
import multiprocessing as mp


class TarWriter:
    def __init__(self, tar_file):
        if not osp.exists(tar_file):
            fd = open(tar_file, "wb")
        else:
            fd = open(tar_file, "r+b")

        self.wds_writer = wds.TarWriter(fd)

    def write(self, key, video_bytes, fmt="mp4"):
        self.wds_writer.write({"__key__": key, fmt: video_bytes})


class FileWriter:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def write(self, key, video_bytes, fmt="mp4"):
        output_file = osp.join(self.output_dir, f"{key}.{fmt}")
        with open(output_file, "wb") as f:
            f.write(video_bytes)


def get_writer(writer, output_dir, shard_id):
    if writer == "tar":
        return TarWriter(osp.join(output_dir, f"{shard_id:04d}.tar"))
    elif writer == "file":
        return FileWriter(output_dir)
    else:
        raise ValueError(f"Invalid writer: {writer}")


class BufferedTextWriter:
    def __init__(self, output_file, buffer_size=100):
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.buffer = []
        if not osp.exists(self.output_file):
            self.handler = open(self.output_file, "w")
        else:
            self.handler = open(self.output_file, "a")

        self.lock = mp.Lock()

    def write(self, cur_str):
        self.buffer.append(cur_str)

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        with self.lock:
            self.handler.write("\n".join(self.buffer) + "\n")
        self.buffer = []
