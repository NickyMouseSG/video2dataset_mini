from dataclasses import dataclass
from ffmpy import FFprobe, FFmpeg
import tempfile
import os, os.path as osp
import io
from decord import VideoReader

from typing import Union
from datetime import datetime, timedelta
import ffmpeg
from typing import Any, List, Tuple, Dict, Literal
import glob
import copy
import cv2
import numpy as np
from pprint import pprint
from loguru import logger
from scenedetect import detect, AdaptiveDetector

from kn_util.data.video import (
    get_frame_indices,
    fill_temporal_param,
    probe_meta,
    save_video_ffmpeg,
    cut_video_clips,
    parse_timestamp_to_secs,
)
from kn_util.utils.system import run_cmd


class ComposedProcessor:
    def __init__(self, processors):
        self.processors = processors

    def __call__(self, byte_stream, meta):
        for processor in self.processors:
            if isinstance(byte_stream, list):
                byte_streams = []
                metas = []
                for bs, m in zip(byte_stream, meta):
                    bs, m = processor(bs, m)

                    if isinstance(bs, list):
                        byte_streams.extend(bs)
                        metas.extend(m)
                    else:
                        byte_streams.append(bs)
                        metas.append(m)
                byte_stream = byte_streams
                meta = metas
            else:
                byte_stream, meta = processor(byte_stream, meta)
        return byte_stream, meta


def get_processor(process_args, media="video"):
    if media == "video":
        processors = [FFmpegProcessor(process_args=process_args)]
        # if process_args.scene_detect:
        #     processors = [SceneCutProcessor()] + processors
        if process_args.clipping:
            processors = [ClippingProcessor()] + processors

        processor = ComposedProcessor(processors)

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

    scene_detect: bool = False
    clipping: bool = False
    timestamp_type: str = "datetime"


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

    def __call__(self, byte_stream, meta):
        f = tempfile.NamedTemporaryFile(suffix=".mp4")
        f.write(byte_stream)

        output_kwargs = []

        try:
            video_meta = probe_meta(f.name)
        except Exception as e:
            raise ValueError(f"Failed to get video meta: {f.name}, {e}")

        if self.args.size is not None:
            size = video_meta["width"], video_meta["height"]

            target_size = _get_target_size(
                self.args.size,
                orig_shape=size,
                resize_mode=self.args.resize_mode,
                divided_by=2,
            )
            output_kwargs.append(f"-vf scale={target_size[0]}:{target_size[1]}")

            video_meta["width"], video_meta["height"] = target_size

        if self.args.fps is not None:
            output_kwargs.append(f"-r {self.args.fps}")
            video_meta["fps"] = self.args.fps

        if self.args.crf is not None:
            output_kwargs.append(f"-crf {self.args.crf}")
            video_meta["crf"] = self.args.crf

        output_f = tempfile.NamedTemporaryFile(suffix=".mp4")
        ff = FFmpeg(
            inputs={f.name: None},
            outputs={output_f.name: " ".join(output_kwargs)},
            global_options=" ".join(
                [
                    "-hide_banner",
                    "-loglevel error",
                    "-y",
                    f"-threads {self.num_threads}",
                ]
            ),
        )
        popen_output = run_cmd(ff.cmd)
        popen_output.check_returncode()

        video_bytes = output_f.read()
        meta.update(video_meta)

        f.close()
        output_f.close()

        return video_bytes, meta


class CV2Processor:
    def __init__(self, process_args: ImageProcessArgs):
        self.args = process_args

    def __call__(self, byte_stream, meta):
        nparr = np.fromstring(byte_stream, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
        target_size = _get_target_size(
            size=self.args.size,
            orig_shape=img_np.shape,
            resize_mode=self.args.resize_mode,
        )
        img_np = cv2.resize(img_np, target_size[::-1], interpolation=cv2.INTER_LINEAR)
        byte_stream = cv2.imencode(".jpg", img_np)[1].tostring()

        return byte_stream, meta


def parse_timesteps(timestamps):

    ret_timesteps = []
    for idx, line_split in enumerate(timestamps):
        st, ed = map(parse_timestamp_to_secs, line_split)
        ret_timesteps += [(st, ed)]

    return ret_timesteps


class ClippingProcessor:
    def __init__(self, with_audio=False):
        self.with_audio = with_audio

    def __call__(self, byte_stream, meta):

        f = tempfile.NamedTemporaryFile(suffix=".mp4")
        f.write(byte_stream)

        timestamps = parse_timesteps(meta["<TIMESTAMP>"])

        byte_streams = []
        metas = []

        with tempfile.TemporaryDirectory() as tmpdir:
            cut_video_clips(
                video_path=f.name,
                timestamps=timestamps,
                output_dir=tmpdir,
                with_audio=self.with_audio,
            )
            mp4_files = sorted(glob.glob(osp.join(tmpdir, "*")))
            for i, mp4_file in enumerate(mp4_files):
                byte_streams.append(open(mp4_file, "rb").read())
                meta_copy = copy.deepcopy(meta)
                meta_copy["<TIMESTAMP>"] = "-".join(map(str, timestamps[i]))
                meta_copy["<SEGMENT>"] = i
                metas += [meta_copy]

        f.close()

        return byte_streams, metas


class SceneCutProcessor:
    def __init__(self, threshold=3.5, min_scene_duration=2.0, num_threads=4):
        self.threshold = threshold
        self.num_threads = num_threads
        self.min_scene_duration = min_scene_duration

    def split_by_ffmpeg(self, video_path, st, ed, fps):

        duration = ed - st
        st = str(timedelta(seconds=st / fps, microseconds=1))[:-3]
        ed = str(timedelta(seconds=ed / fps, microseconds=1))[:-3]
        # duration = str(timedelta(seconds=duration / fps, microseconds=1))[:-3]

        output_f = tempfile.NamedTemporaryFile(suffix=".mp4")
        ff = FFmpeg(
            inputs={video_path: None},
            outputs={output_f.name: f"-ss {st} -to {ed}"},
            global_options=f"-hide_banner -loglevel error -y -threads {self.num_threads}",
        )
        popen_output = run_cmd(ff.cmd)
        popen_output.check_returncode()

        byte_stream = output_f.read()
        output_f.close()
        return byte_stream

    def split_by_decord(self, vr, st, ed, fps):
        output_f = tempfile.NamedTemporaryFile(suffix=".mp4")
        frames = vr.get_batch(list(range(st, ed))).asnumpy()
        save_video_ffmpeg(frames, output_f.name, crf=0, fps=fps)
        return output_f.read()

    def __call__(self, byte_stream, meta):

        f = tempfile.NamedTemporaryFile(suffix=".mp4")
        f.write(byte_stream)

        cap = cv2.VideoCapture(f.name)
        fps = cap.get(cv2.CAP_PROP_FPS)

        detector = AdaptiveDetector(adaptive_threshold=self.threshold)
        scene_list = detect(f.name, detector=detector, start_in_scene=True)

        meta["<SCENECUT>"] = scene_list

        return byte_stream, meta
