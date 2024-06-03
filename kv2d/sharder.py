import pyarrow as pa
from kn_util.utils.multiproc import map_async_with_thread
import pyarrow.csv as pa_csv
import pyarrow.feather as pa_feather
import pandas as pd
import os
from loguru import logger
from dataclasses import dataclass


@dataclass
class ReadArguments:
    headers: bool = True
    url_col: str = "url"
    id_col: str = "id"


class Sharder:
    # simiplifed version of the original InputSharder class
    # https://github.com/iejMac/video2dataset/blob/main/video2dataset/input_sharder.py

    def __init__(
        self,
        input_file,
        read_args: ReadArguments,
        shard_size=1000,
        shard_dir=".shards",
        rank_id=0,
        world_size=1,
    ):
        self.read_args = read_args
        df = self.read(input_file, read_args)

        self._row_count = len(df)
        os.makedirs(shard_dir, exist_ok=True)

        self.rank_id = rank_id
        self.world_size = world_size

        self.shard_files = self.write_shards(df, shard_size=shard_size, shard_dir=shard_dir)

    def read_csv(self, input_file, delimiter=",", headers=True):
        if isinstance(headers, bool) and headers:
            with open(input_file, "r") as f:
                headers = f.readline().strip().split(delimiter)
        elif isinstance(headers, list):
            pass
        else:
            raise ValueError("headers must be a list or True")

        # https://stackoverflow.com/questions/78056946/how-to-read-a-huge-csv-faster
        # allow the block_size to cover the longest line
        megabyte = 1<<20
        max_len = max(len(line) for line in open(input_file, 'rb'))
        block_size = megabyte * (1 + (max_len-1) // megabyte)  

        return pa_csv.read_csv(
            input_file,
            read_options=pa_csv.ReadOptions(column_names=headers, block_size=block_size),
            parse_options=pa_csv.ParseOptions(delimiter=delimiter),
        )

    def read(self, input_file, read_args):
        file_format = input_file.split(".")[-1]
        if file_format in ["tsv"]:
            delimiter = "," if file_format == "csv" else "\t"
            df = self.read_csv(
                input_file,
                delimiter=delimiter,
                headers=read_args.headers,
            )
            return df
        elif file_format in ["csv"]:
            df = pa.Table.from_pandas(pd.read_csv(input_file))
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

        num_shard_per_rank = (num_shards + self.world_size - 1) // self.world_size

        # local_shard_ids is a list of shard_ids that are assigned to the current rank
        # each element should be a global shard index
        local_shard_ids = [i for i in range(self.rank_id * num_shard_per_rank, (self.rank_id + 1) * num_shard_per_rank) if i < num_shards]
        self.local_shard_ids = local_shard_ids
        shard_spans = [(i * shard_size, min((i + 1) * shard_size, len(df))) for i in local_shard_ids]
        local_df_shards = [df.slice(start, end - start + 1) for start, end in shard_spans]

        # prevent slice df in multiple threads, not safe
        shard_files = map_async_with_thread(
            iterable=list(zip(local_shard_ids, local_df_shards)),
            func=lambda x: write_shard(
                df_shard=x[1],
                shard_id=x[0],
                shard_dir=shard_dir,
            ),
            verbose=False,
        )
        logger.info(f"Input data ({len(df)} rows) has been sharded into {len(local_shard_ids)} shards in rank {self.rank_id}.")

        return shard_files

    def get_global_id(self, shard_id):
        return self.local_shard_ids[shard_id]

    def get_local_id(self, global_id):
        return self.local_shard_ids.index(global_id)

    def fetch_shards(self, local_shard_ids):
        if len(local_shard_ids) == 0:
            []
        if len(local_shard_ids) == 1:
            global_idx = self.get_global_id(local_shard_ids[0])
            return [self.fetch_shard(local_shard_ids[0])], [global_idx]
        else:
            global_idxs = [self.get_global_id(i) for i in local_shard_ids]
            ret = map_async_with_thread(
                iterable=local_shard_ids,
                func=self.fetch_shard,
                verbose=False,
            )
            return ret, global_idxs

    @property
    def row_count(self):
        return self._row_count

    def fetch_shard(self, local_shard_id):
        url_col = self.read_args.url_col
        id_col = self.read_args.id_col
        shard_df = pa_feather.read_table(self.shard_files[local_shard_id])
        column_names = set(shard_df.column_names)
        column_names.remove(url_col)
        column_names.remove(id_col)
        shard_size = len(shard_df)

        url = shard_df[url_col].to_pylist()
        vid = shard_df[id_col].to_pylist()
        if self.read_args.headers:
            url = url[1:]
            vid = vid[1:]

        meta = [{k: shard_df[k][i] for k in column_names} for i in range(shard_size)]

        return (url, vid, meta)

    def __getitem__(self, shard_id):
        return self.fetch_shard(shard_id)

    def __len__(self):
        return len(self.shard_files)
