import argparse
from kn_util.utils.io import load_csv
import os
import os.path as osp
from .download import download, ProcessArguments
from .sharder import Sharder, ReadArguments


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, required=True, help="Input file containing video URLs"
    )
    parser.add_argument(
        "--num_processes", type=int, default=16, help="Number of processes to use"
    )
    parser.add_argument(
        "--num_threads", type=int, default=16, help="Number of threads to use"
    )
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
    parser.add_argument(
        "--log_file", type=str, default="video_downloader.log", help="Log file"
    )

    # Media Arguments
    parser.add_argument(
        "--media", type=str, default="video", help="Media type to download"
    )

    # Sharder Arguments
    parser.add_argument(
        "--url_col", type=str, default="url", help="Column name for video URLs"
    )
    parser.add_argument(
        "--id_col", type=str, default="id", help="Column name for video IDs"
    )

    # Process Arguments
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="All processing arguments will be ignored",
    )
    parser.add_argument(
        "--fps", type=int, default=8, help="Frames per second for the output video"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Size of the smaller dimension of the output video",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=None,
        help="Maximum size of the smaller dimension of the output video",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Center crop the video to the specified size",
        default=False,
    )
    parser.add_argument(
        "--quality", type=int, default=5, help="Quality of the output video"
    )
    parser.add_argument(
        "--crf", type=int, default=23, help="CRF value for the output video"
    )

    # Writer Arguments
    parser.add_argument(
        "--writer", type=str, default="file", help="Writer to use for saving videos"
    )

    # Multi-Node Arguments
    parser.add_argument("--rank", type=int, default=0, help="Rank of the current node")
    parser.add_argument(
        "--world_size", type=int, default=1, help="Total number of nodes"
    )

    # Upload Arguments
    parser.add_argument(
        "--upload_hf",
        action="store_true",
        help="Upload the downloaded videos to a cloud storage",
        default=False,
    )
    parser.add_argument(
        "--repo_id", type=str, help="Repository ID for the cloud storage"
    )
    parser.add_argument(
        "--upload_s3",
        action="store_true",
        help="Upload the downloaded videos to an S3 bucket",
        default=False,
    )
    parser.add_argument("--bucket", type=str, help="Bucket name for the S3 storage")
    parser.add_argument(
        "--delete_local",
        action="store_true",
        help="Delete the local videos after uploading",
        default=False,
    )

    # Debug Arguments
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the download process",
        default=False,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run the download process",
        default=False,
    )

    return parser.parse_known_args()[0]


def main():
    from kn_util.utils.logger import setup_logger_loguru

    args = get_args()

    setup_logger_loguru(filename=args.log_file)

    os.makedirs(osp.join(args.output_dir, ".meta"), exist_ok=True)
    os.makedirs(osp.dirname(args.log_file), exist_ok=True)

    read_args = ReadArguments(headers=True, url_col=args.url_col, id_col=args.id_col)
    sharder = Sharder(
        input_file=args.input_file,
        read_args=read_args,
        shard_size=args.shard_size,
        shard_dir=osp.join(args.output_dir, ".shards"),
        rank_id=args.rank,
        world_size=args.world_size,
    )

    process_args = ProcessArguments(
        disabled=args.download_only,
        fps=args.fps,
        size=args.size,
        max_size=args.max_size,
        center_crop=args.center_crop,
        quality=args.quality,
        crf=args.crf,
    )
    download(
        media=args.media,
        sharder=sharder,
        rank=args.rank,
        process_args=process_args,
        writer=args.writer,
        output_dir=args.output_dir,
        num_processes=args.num_processes,
        num_threads=args.num_threads,
        semaphore_limit=args.semaphore_limit,
        max_retries=args.max_retries,
        # upload
        upload_hf=args.upload_hf,
        repo_id=args.repo_id,
        upload_s3=args.upload_s3,
        bucket=args.bucket,
        delete_local=args.delete_local,
        # debug
        profile=args.profile,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
