import webdataset as wds
import os, os.path as osp
import numpy as np
import multiprocessing as mp
import tempfile
import numpy as np
import torch
import time
from loguru import logger
from PIL import Image
import json
from dataclasses import dataclass

from .utils import safe_open

from kn_util.utils.system import run_cmd, force_delete_dir
from kn_util.data.video import save_video_imageio, array_to_video_bytes, save_video_ffmpeg


@dataclass
class UploadArgs:
    upload_hf: bool = False
    repo_id: str = ""
    upload_s3: bool = False
    endpoint_url: str = ""
    bucket: str = ""
    delete_local: bool = False


def is_tensor(x):
    return isinstance(x, torch.Tensor) or isinstance(x, np.ndarray)


class TarWriter:
    def __init__(self, tar_file, media="video"):
        fd = open(tar_file, "wb")
        self.media = media
        self.wds_writer = wds.TarWriter(fd)

    def write(self, key, video_bytes, fmt="mp4", video_meta=None):
        video_bytes = self.maybe_frames_to_bytes(video_bytes)
        self.wds_writer.write({"__key__": key, fmt: video_bytes})

    def maybe_frames_to_bytes(self, frames):
        if isinstance(frames, torch.Tensor) or isinstance(frames, np.ndarray):
            video_bytes = array_to_video_bytes(frames)
        return video_bytes

    @property
    def downloaded_ids(self):
        return set()  # cannot resume vids from tar file

    def close(self):
        self.wds_writer.close()


class FileWriter:
    def __init__(self, cache_dir, process_args, upload_args, media="video"):
        self.cache_dir = cache_dir
        self.media = media
        os.makedirs(self.cache_dir, exist_ok=True)
        self.process_args = process_args
        self.upload_args = upload_args

    def upload(self):
        raise NotImplementedError

    def write(self, key, array, fmt="mp4", video_meta=None):
        assert isinstance(array, (bytearray, bytes)), f"Invalid array type: {type(array)}"
        if isinstance(array, (bytearray, bytes)):
            with open(osp.join(self.cache_dir, f"{key}.{fmt}"), "wb") as f:
                f.write(array)
            return

        # ! Deprecated
        if False:
            cache_file = osp.join(self.cache_dir, f"{key}.{fmt}")
            with tempfile.NamedTemporaryFile(suffix="." + fmt, delete=False) as f:
                if self.media == "video":
                    save_video_ffmpeg(array, f.name, fps=video_meta["fps"], crf=self.process_args.crf)
                    # save_video_imageio(array, f.name, fps=video_meta["fps"], quality=self.quality)
                elif self.media == "image":
                    Image.fromarray(array).save(f.name)
                else:
                    raise ValueError(f"Invalid media type: {self.media}")
                run_cmd(f"mv {f.name} {cache_file}")

    @property
    def downloaded_ids(self):
        cache_files = set(os.listdir(self.cache_dir))
        downloaded_ids = set()
        for cache_file in cache_files:
            key, _ = cache_file.split(".")
            downloaded_ids.add(key)

        return downloaded_ids

    def close(self):
        pass


class CachedTarWriter(FileWriter):
    def __init__(self, tar_file, cache_dir, process_args, upload_args, media="video"):
        super().__init__(cache_dir=cache_dir, media=media, process_args=process_args, upload_args=upload_args)
        self.tar_file = tar_file

    def upload(self):
        try:
            popens = []
            if self.upload_args.upload_hf:
                # while osp.exists(f"~repo"):
                #     time.sleep(1)

                # run_cmd(f"GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/{self.upload_args.repo_id} ~repo", async_cmd=False)
                # run_cmd(f"cp {self.tar_file} ~repo", async_cmd=False)
                # run_cmd(
                #     f"cd ~repo && git lfs track {tar_filename} && git add {tar_filename} && git commit --amend -m 'add files' && git push -f",
                #     async_cmd=False,
                # )
                # run_cmd(f"rm -rf ~repo", async_cmd=False)
                # from huggingface_hub.hf_api import HfApi
                # hf = HfApi()
                # hf.upload_file(path_in_repo=tar_filename, path_or_fileobj=self.tar_file, repo_id=self.upload_args.repo_id, repo_type="dataset")
                tar_filename = osp.basename(self.tar_file)
                cmd = f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload --repo-type dataset {self.upload_args.repo_id} {self.tar_file}"
                logger.info(cmd)
                popen = run_cmd(cmd, async_cmd=True)
                popens += [popen]
                logger.info(f"{self.tar_file} uploaded to {self.upload_args.repo_id}")

            if self.upload_args.upload_s3:
                popen = run_cmd(
                    f"aws s3 cp {self.tar_file} s3://sg-sail-home-wangjing/home/{self.upload_args.bucket} --endpoint-url {self.upload_args.endpoint_url}",
                    async_cmd=True,
                )
                popens += [popen]
                logger.info(f"{self.tar_file} uploaded to {self.upload_args.bucket} via S3")

            for popen in popens:
                popen.wait()

            if self.upload_args.delete_local:
                run_cmd(f"rm -rf {self.tar_file}", async_cmd=False)
                logger.info(f"{self.tar_file} deleted")
        except:
            logger.error(f"Failed to upload {self.tar_file}")

    def close(self):
        cur_dir = osp.dirname(self.tar_file)
        key_file = osp.join(cur_dir, "keys.jsonl")
        tar_filename = osp.basename(self.tar_file)

        writer = wds.TarWriter(open(self.tar_file, "wb"))
        keys = []

        cache_files = sorted(os.listdir(self.cache_dir))
        for cache_file in cache_files:
            with open(osp.join(self.cache_dir, cache_file), "rb") as f:
                video_bytes = f.read()
                key, fmt = cache_file.split(".")
                writer.write({"__key__": key, fmt: video_bytes})
                keys.append(key)

        self.upload()
        run_cmd(f"rm -rf {self.cache_dir}", async_cmd=False)

        with safe_open(key_file, "a") as f:
            item = {"tar": tar_filename, "keys": keys}
            json_str = json.dumps(item)
            f.write(json_str + "\n")


def get_writer(writer, output_dir, shard_id, process_args, upload_args, media="video"):
    from .utils import logits

    cache_dir = osp.join(output_dir, f"cache.{shard_id:0{logits}d}")
    tar_file = osp.join(output_dir, f"{shard_id:0{logits}d}.tar")
    if writer == "file":
        logger.warning("Using file writer, upload_args will be ignored")
        return FileWriter(cache_dir=cache_dir, process_args=process_args, upload_args=upload_args, media=media)
    elif writer == "tar":
        return CachedTarWriter(tar_file, cache_dir=cache_dir, media=media, process_args=process_args, upload_args=upload_args)
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
