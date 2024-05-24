import argparse
from kn_util.utils.io import load_csv
import os
import os.path as osp
from .download import download, ProcessArguments
from .sharder import Sharder, ReadArguments


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file containing video URLs")
    parser.add_argument("--num_processes", type=int, default=16, help="Number of processes to use")
    parser.add_argument("--num_threads", type=int, default=16, help="Number of threads to use")
    parser.add_argument("--output_dir", type=str, help="Output directory to save videos", default="videos")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for failed downloads")
    parser.add_argument("--shard_size", type=int, default=1000, help="Number of videos to download in each shard")
    parser.add_argument("--semaphore_limit", type=int, default=16, help="Maximum number of downloads accumulating in thread")
    parser.add_argument("--log_file", type=str, default="video_downloader.log", help="Log file")

    # Sharder Arguments
    parser.add_argument("--url_col", type=str, default="url", help="Column name for video URLs")
    parser.add_argument("--vid_col", type=str, default="vid", help="Column name for video IDs")

    # Process Arguments
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the output video")
    parser.add_argument("--size", type=int, default=512, help="Size of the smaller dimension of the output video")
    parser.add_argument("--max_size", type=int, default=None, help="Maximum size of the smaller dimension of the output video")

    # Writer Arguments
    parser.add_argument("--writer", type=str, default="file", help="Writer to use for saving videos")

    # Multi-Node Arguments
    parser.add_argument("--rank", type=int, default=0, help="Rank of the current node")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of nodes")

    # Upload Arguments
    parser.add_argument("--upload", action="store_true", help="Upload videos to Hugging Face repository")
    parser.add_argument("--delete_local", action="store_true", help="Delete videos after uploading")
    parser.add_argument("--hf_repo", type=str, default=None, help="Hugging Face repository to upload videos to")

    return parser.parse_known_args()[0]


def main():
    from kn_util.utils.logger import setup_logger_loguru

    args = get_args()

    setup_logger_loguru(filename=args.log_file)

    os.makedirs(osp.join(args.output_dir, ".meta"), exist_ok=True)

    read_args = ReadArguments(headers=True, url_col=args.url_col, vid_col=args.vid_col)
    sharder = Sharder(
        input_file=args.input_file,
        read_args=read_args,
        shard_size=args.shard_size,
        shard_dir=osp.join(args.output_dir, ".shards"),
        rank_id=args.rank,
        world_size=args.world_size,
    )

    process_args = ProcessArguments(fps=args.fps, size=args.size, max_size=args.max_size)
    download(
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
        upload=args.upload,
        repo_id=args.hf_repo,
        delete_local=args.delete_local,
        # debug
        # profile=True,
    )


if __name__ == "__main__":
    main()
