logits=4

def shardid2name(shard_id):
    return f"{shard_id:0{logits}d}"