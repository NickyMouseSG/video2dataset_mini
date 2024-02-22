import pyarrow as pa
from kn_util.utils.multiproc import map_async_with_thread
import pyarrow.csv as pa_csv
import pyarrow.feather as pa_feather
import os
from loguru import logger


class Sharder:
    # simiplifed version of the original InputSharder class
    # https://github.com/iejMac/video2dataset/blob/main/video2dataset/input_sharder.py

    def __init__(
        self,
        input_file,
        read_kwargs,
        shard_size=1000,
        shard_dir=".shards",
    ):
        self.read_kwargs = read_kwargs
        df = self.read(input_file, **read_kwargs)
        self._row_count = len(df)
        os.makedirs(shard_dir, exist_ok=True)
        self.shard_files = self.write_shards(df, shard_size=shard_size, shard_dir=shard_dir)

    def read_csv(self, input_file, delimiter=",", headers=True):
        if isinstance(headers, bool) and headers:
            with open(input_file, "r") as f:
                headers = f.readline().strip().split(delimiter)
        elif isinstance(headers, list):
            pass
        else:
            raise ValueError("headers must be a list or True")

        return pa_csv.read_csv(
            input_file,
            read_options=pa_csv.ReadOptions(column_names=headers),
            parse_options=pa_csv.ParseOptions(delimiter=delimiter),
        )

    def read(self, input_file, **kwargs):
        file_format = input_file.split(".")[-1]
        if file_format in ["csv", "tsv"]:
            delimiter = "," if file_format == "csv" else "\t"
            df = self.read_csv(
                input_file,
                delimiter=delimiter,
                headers=kwargs.get("headers", True),
            )
            return df
        else:
            raise NotImplementedError(f"File format {file_format} not supported")

    def write_shards(self, df, shard_size=1000, shard_dir="."):
        num_shards = (len(df) + shard_size - 1) // shard_size
        num_logits = len(str(num_shards))

        def write_shard(df_shard, shard_id, shard_dir):
            shard_file = f"{shard_dir}/shard_{shard_id:0{num_logits}d}.parquet"
            # original code use pa.ipc.new_file here, not sure if it's a bug or not
            pa_feather.write_feather(df_shard, shard_file)

            return shard_file

        shard_spans = [(i * shard_size, min((i + 1) * shard_size, len(df))) for i in range(num_shards)]
        df_shards = [df.slice(start, end - start) for start, end in shard_spans]

        # prevent slice df in multiple threads, not safe
        shard_files = map_async_with_thread(
            iterable=list(range(num_shards)),
            func=lambda shard_id: write_shard(
                df_shard=df_shards[shard_id],
                shard_id=shard_id,
                shard_dir=shard_dir,
            ),
            verbose=False,
        )
        logger.info(f"Input data has been sharded into {num_shards} shards")

        return shard_files

    def fetch_shards(self, shard_ids):
        if len(shard_ids) == 0:
            []
        if len(shard_ids) == 1:
            return [self.fetch_shard(shard_ids[0])]
        else:
            ret = map_async_with_thread(
                iterable=shard_ids,
                func=self.fetch_shard,
                verbose=False,
            )
            return ret

    @property
    def row_count(self):
        return self._row_count

    def fetch_shard(self, shard_id):
        url_col = self.read_kwargs.get("url_column", "url")
        vid_col = self.read_kwargs.get("vid_column", "vid")
        shard_df = pa_feather.read_table(self.shard_files[shard_id])
        column_names = set(shard_df.column_names)
        column_names.remove(url_col)
        column_names.remove(vid_col)
        shard_size = len(shard_df)

        url = shard_df[url_col].to_pylist()
        vid = shard_df[vid_col].to_pylist()
        meta = [{k: shard_df[k][i] for k in column_names} for i in range(shard_size)]

        return (url, vid, meta)

    def __getitem__(self, shard_id):
        return self.fetch_shard(shard_id)

    def __len__(self):
        return len(self.shard_files)
