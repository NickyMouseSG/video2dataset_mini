from dataclasses import dataclass
from ffmpy import FFprobe, FFmpeg
import tempfile
import os, os.path as osp
import io
from decord import VideoReader

from typing import Union
from datetime import datetime
import ffmpeg
from typing import Any, List, Tuple, Dict, Literal
import glob
import copy
import cv2
import numpy as np
from pprint import pprint
from loguru import logger

from kn_util.data.video import get_frame_indices, fill_temporal_param, probe_meta
from kn_util.utils.system import run_cmd


def get_processor(process_args, media="video"):
    if media == "video":
        processor = FFmpegProcessor(process_args=process_args)
    elif media == "image":
        processor = CV2Processor(process_args=process_args)
    else:
        raise ValueError(f"Invalid media type: {media}")
    return processor


def _get_target_size(size, orig_shape, resize_mode="shortest", divided_by=1):
    cmp = lambda x, y: x < y if resize_mode == "shortest" else x > y
    if cmp(orig_shape[0], orig_shape[1]):
        shape = [size, int(size * orig_shape[1] / orig_shape[0])]
    else:
        shape = [int(size * orig_shape[0] / orig_shape[1]), size]

    shape[0] = shape[0] // divided_by * divided_by
    shape[1] = shape[1] // divided_by * divided_by

    return tuple(shape)


@dataclass
class ImageProcessArgs:
    skip_process: bool = False
    size: int = None
    max_size: int = None
    center_crop: bool = False
    resize_mode: str = "shortest"
    center_crop: bool = False


@dataclass
class VideoProcessArgs(ImageProcessArgs):
    fps: int = None
    crf: int = None


# class DecordProcessor:
#     def __init__(self, process_args: VideoProcessArgs):
#         self.args = process_args

#     def get_video_meta(self, byte_stream):
#         byte_io = io.BytesIO(byte_stream)
#         reader = VideoReader(byte_io, num_threads=4)
#         shape = reader[0].shape
#         shape = (shape[1], shape[0])
#         fps = reader.get_avg_fps()
#         frame_count = len(reader)

#         return {
#             "shape": shape,
#             "fps": fps,
#             "frame_count": frame_count,
#             "duration": frame_count / fps,
#         }

#     def __call__(self, byte_stream, timestamps):
#         # 1. prepare the video reader
#         byte_io = io.BytesIO(byte_stream)
#         video_meta = self.get_video_meta(byte_stream)
#         orig_shape = video_meta["shape"]
#         fps = video_meta["fps"]

#         target_size = _get_target_size(byte_io, orig_shape=orig_shape, resize_mode=self.args.resize_mode)
#         vr = VideoReader(byte_stream, height=target_size[0], width=target_size[1], num_threads=4)

#         # 2. calculate frame indices
#         for st, ed in timestamps:
#             clip_duration = ed - st

#             duration_tgt, num_frames_tgt, fps_tgt = fill_temporal_param(duration=duration, fps=self.args.fps)


#         # TODO - implement the rest of the function


class FFmpegProcessor:
    def __init__(self, process_args: VideoProcessArgs, num_threads=4):
        self.args = process_args
        self.num_threads = num_threads

    def get_size(self, path):
        size = None
        video_meta = probe_meta(path)

        if video_meta is None:
            raise ValueError(f"Failed to get video meta: {path}, seems like a corrupted video file")

        size = video_meta["width"], video_meta["height"]

        return size

    def __call__(self, byte_stream):
        f = tempfile.NamedTemporaryFile(suffix=".mp4")
        f.write(byte_stream)

        output_kwargs = []

        if self.args.size is not None:
            size = self.get_size(f.name)

            target_size = _get_target_size(
                self.args.size,
                orig_shape=size,
                resize_mode=self.args.resize_mode,
                divided_by=2,
            )
            output_kwargs.append(f"-vf scale={target_size[0]}:{target_size[1]}")

        if self.args.fps is not None:
            output_kwargs.append(f"-r {self.args.fps}")

        if self.args.crf is not None:
            output_kwargs.append(f"-crf {self.args.crf}")

        output_f = tempfile.NamedTemporaryFile(suffix=".mp4")
        ff = FFmpeg(
            inputs={f.name: None},
            outputs={output_f.name: " ".join(output_kwargs)},
            global_options=" ".join(["-hide_banner", "-loglevel error", "-y", f"-threads {self.num_threads}"]),
        )
        popen_output = run_cmd(ff.cmd)
        popen_output.check_returncode()
        video_bytes = output_f.read()

        f.close()
        output_f.close()

        return video_bytes


class CV2Processor:
    def __init__(self, process_args: ImageProcessArgs):
        self.args = process_args

    def __call__(self, byte_stream):
        nparr = np.fromstring(byte_stream, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
        target_size = _get_target_size(
            size=self.args.size,
            orig_shape=img_np.shape,
            resize_mode=self.args.resize_mode,
        )
        img_np = cv2.resize(img_np, target_size[::-1], interpolation=cv2.INTER_LINEAR)
        byte_stream = cv2.imencode(".jpg", img_np)[1].tostring()

        return byte_stream
