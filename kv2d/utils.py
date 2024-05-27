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
                # Use os.open to atomically create the lock file
                self.fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break  # Successfully created lock file, exit loop
            except FileExistsError:
                time.sleep(3)  # Lock file exists, wait

        self.f = open(self.filename, self.mode)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the file
        self.f.close()
        # Close the file descriptor for the lock file
        os.close(self.fd)
        # Remove the lock file
        os.remove(self.lock_file)
        # Returning False will propagate any exceptions
        return False
