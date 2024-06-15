import argparse
from kn_util.utils.io import load_csv
import os
import os.path as osp
from loguru import logger

from .download import download
from .sharder import Sharder, ReadArguments
from .process import VideoProcessArgs, ImageProcessArgs
from .writer import UploadArgs

from kn_util.utils.system import run_cmd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file containing video URLs")
    parser.add_argument("--num_processes", type=int, default=16, help="Number of processes to use")
    parser.add_argument("--num_threads", type=int, default=16, help="Number of threads to use")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to save videos",
        default="videos",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed downloads",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1000,
        help="Number of videos to download in each shard",
    )
    parser.add_argument(
        "--semaphore_limit",
        type=int,
        default=16,
        help="Maximum number of downloads accumulating in thread",
    )
    parser.add_argument("--log_file", type=str, default="downloader.log", help="Log file")

    # Media Arguments
    parser.add_argument("--media", type=str, default="video", help="Media type to download")

    # Sharder Arguments
    parser.add_argument("--url_col", type=str, default="url", help="Column name for video URLs")
    parser.add_argument("--id_col", type=str, default="id", help="Column name for video IDs")
    parser.add_argument("--timestamp_col", type=str, default=None, help="Column name for timestamps")
    parser.add_argument("--include_shard_ids", nargs="+", default=None, help="Shard IDs to include")

    # Process Arguments
    parser.add_argument("--skip_process", action="store_true", help="All processing arguments will be ignored")
    parser.add_argument("--process_download", action="store_true", help="Process videos while downloading")
    parser.add_argument("--size", type=int, default=None, help="Size of the smaller dimension of the output video")
    parser.add_argument("--max_size", type=int, default=None, help="Maximum size of the smaller dimension of the output video")
    parser.add_argument("--resize_mode", type=str, default="shortest", help="Mode for resizing the video")
    parser.add_argument("--crf", type=int, default=None, help="CRF value for the output video")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the output video")

    # Writer Arguments
    parser.add_argument("--writer", type=str, default="file", help="Writer to use for saving videos")

    # Multi-Node Arguments
    parser.add_argument("--rank_split", type=int, default=1, help="Number of ranks to split the input file into")
    parser.add_argument("--rank", type=int, default=0, help="Rank ID for the current node")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of nodes")

    # Upload Arguments
    parser.add_argument("--upload_hf", action="store_true", help="Upload the downloaded videos to a cloud storage", default=False)
    parser.add_argument("--repo_id", type=str, help="Repository ID for the cloud storage")
    parser.add_argument("--upload_s3", action="store_true", help="Upload the downloaded videos to an S3 bucket", default=False)
    parser.add_argument("--bucket", type=str, help="Path to the S3 bucket")
    parser.add_argument("--endpoint_url", type=str, help="Endpoint URL for the S3 bucket")
    parser.add_argument("--delete_local", action="store_true", help="Delete the local videos after uploading", default=False)

    # Debug Arguments
    parser.add_argument("--profile", action="store_true", help="Profile the download process", default=False)
    parser.add_argument("--debug", action="store_true", help="Dry run the download process", default=False)

    return parser.parse_known_args()[0]


def get_process_args(args):
    if args.media == "video":
        process_args = VideoProcessArgs(
            skip_process=args.skip_process,
            size=args.size,
            max_size=args.max_size,
            resize_mode=args.resize_mode,
            crf=args.crf,
            fps=args.fps,
        )
    elif args.media == "image":
        process_args = ImageProcessArgs(
            skip_process=args.skip_process,
            size=args.size,
            max_size=args.max_size,
            resize_mode=args.resize_mode,
        )
    else:
        raise ValueError(f"Unknown media type: {args.media}")

    return process_args


def get_upload_args(args):
    if args.upload_hf:
        try:
            import hf_transfer
        except:
            raise ImportError("Please install hf-transfer to use the Hugging Face upload feature")

    return UploadArgs(
        upload_hf=args.upload_hf,
        repo_id=args.repo_id,
        upload_s3=args.upload_s3,
        bucket=args.bucket,
        endpoint_url=os.environ.get("S3_ENDPOINT_URL", args.endpoint_url),
        delete_local=args.delete_local,
    )


def main_per_rank(args):

    # if args.timestamp_col is not None:
    #     logger.info("Forced to set args.download_only and args.process_download to True when using timestamp_col")
    #     args.download_only = True
    #     args.process_download = True

    os.makedirs(osp.join(args.output_dir, ".meta"), exist_ok=True)
    os.makedirs(osp.dirname(osp.abspath(args.log_file)), exist_ok=True)

    read_args = ReadArguments(
        headers=True,
        url_col=args.url_col,
        id_col=args.id_col,
        timestamp_col=args.timestamp_col,
        include_shard_ids=args.include_shard_ids,
    )
    sharder = Sharder(
        input_file=args.input_file,
        read_args=read_args,
        shard_size=args.shard_size,
        shard_dir=osp.join(args.output_dir, ".shards"),
        rank_id=args.rank,
        world_size=args.world_size,
    )

    process_args = get_process_args(args)
    upload_args = get_upload_args(args)

    download(
        media=args.media,
        sharder=sharder,
        rank=args.rank,
        process_args=process_args,
        upload_args=upload_args,
        writer=args.writer,
        output_dir=args.output_dir,
        num_processes=args.num_processes,
        num_threads=args.num_threads,
        semaphore_limit=args.semaphore_limit,
        max_retries=args.max_retries,
        # debug
        profile=args.profile,
        debug=args.debug,
    )


def main():
    from kn_util.utils.logger import setup_logger_loguru

    args = get_args()
    setup_logger_loguru(filename=args.log_file, logger=logger)

    if args.rank_split > 1:
        # this means even single rank of data is too large for storage
        # normally this should be used together with --delete_local
        assert args.delete_local, "rank_split should be used together with --delete_local"

        args.world_size = args.rank_split * args.world_size
        ranks = range(args.rank * args.rank_split, (args.rank + 1) * args.rank_split)

        for rank in ranks:
            args.rank = rank
            args.log_file = args.log_file.format(rank=rank)
            main_per_rank(args)
    else:
        args.rank = int(args.rank)
        args.log_file = args.log_file.format(rank=args.rank)
        main_per_rank(args)


if __name__ == "__main__":
    main()
