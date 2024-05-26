import os, os.path as osp
import time

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
        while osp.exists(self.lock_file):
            time.sleep(3)

        os.system(f"touch {self.lock_file}")
        self.f = open(self.filename, self.mode)
        return self.f

    def __exit__(self, *args):
        os.system(f"rm {self.lock_file}")
        self.f.close()
