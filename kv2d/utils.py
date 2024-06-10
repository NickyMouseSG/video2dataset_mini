import os, os.path as osp
import time
import re
import fcntl

logits = 4


def shardid2name(shard_id):
    return f"{shard_id:0{logits}d}"


class safe_open:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

        cur_dir = osp.dirname(self.filename)
        cur_filename = osp.basename(self.filename)
        self.lock_file = osp.join(cur_dir, f".{cur_filename}.lock")

    def __enter__(self):
        self.f = open(self.filename, self.mode)
        fcntl.lockf(self.f, fcntl.LOCK_EX)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.flush()
        fcntl.lockf(self.f, fcntl.LOCK_UN)
        self.f.close()
        return False


from datetime import datetime, timedelta


# def parse_timestamps(timestamps):
#     st_str, ed_str = timestamps.split("-")

#     def _parse(t_str):
#         if ":" not in t_str:
#             t = timedelta(seconds=int(st_str))
#         elif t_str.count(":") == 1:
#             t = datetime.strptime(t_str, "%M:%S")
#         elif t_str.count(":") == 2:
#             t = datetime.strptime(t_str, "%H:%M:%S")
#         else:
#             raise ValueError(f"Invalid timestamp format: {t_str}")
#         return t

#     st = _parse(st_str)
#     ed = _parse(ed_str)

#     delta = ed - st
#     delta_seconds = delta.total_seconds()

#     return st.strftime("%H:%M:%S"), delta_seconds
