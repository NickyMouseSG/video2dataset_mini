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
        while True:
            try:
                # 使用 os.open 以原子方式创建锁文件
                self.fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break  # 成功创建锁文件，跳出循环
            except FileExistsError:
                time.sleep(3)  # 锁文件存在，等待

        self.f = open(self.filename, self.mode)
        return self.f
