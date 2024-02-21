import argparse
from kn_util.utils.io import load_csv
import os
import os.path as osp
from downloader import VideoDownloader


def add_args(parser):
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
    parser.add_argument("--shard_size", type=int, default=1000, help="Number of videos to download in each shard")
    parser.add_argument("--semaphore_limit", type=int, default=32, help="Maximum number of downloads accumulating in thread")
    parser.add_argument("--log_file", type=str, default="video_downloader.log", help="Log file")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs")


def main():
    from kn_util.utils.logger import setup_logger_loguru

    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_known_args()[0]

    setup_logger_loguru(
        filename=args.log_file,
        include_filepath=False,
        include_function=False,
        stdout=False,
    )

    input_metas = load_csv(args.input_file, delimiter="\t", has_header=True)
    assert "url" in input_metas[0], "Input meta data must contain 'url' field for downloading"
    assert "vid" in input_metas[0], "Input meta data must contain 'vid' field for naming"

    os.makedirs(osp.join(args.output_dir, ".meta"), exist_ok=True)

    urls = []
    for row in input_metas:
        url = row.pop("url")
        urls.append(url)

    video_downloader = VideoDownloader(
        num_processes=args.num_processes,
        num_threads=args.num_threads,
        semaphore_limit=args.semaphore_limit,
        shard_size=args.shard_size,
        max_retries=args.max_retries,
        verbose=args.verbose,
    )

    video_downloader.download(
        urls,
        input_metas,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    progress = None
    main()
