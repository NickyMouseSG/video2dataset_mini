# To-Do

- [ ] Support pyarrow table for sharding meta (.csv does not support too many urls)

# Install kn_util

```bash
pip install git+https://github.com/K-Nick/kn-toolkit
```

# Usage example

```bash
kv2d --input_file url_mini.tsv --num_processes 16 --num_threads 32 --verbose
```

![alt text](assets/image.png)

# Speed Testing

5000 videos, 16 processes, 32 threads
Note that this is not a strict comparison since video2dataset_mini does not involve post processing of downloaded video.

| Lib           | Times  |
| ------------- | ------ |
| kv2d          | 7m25s  |
| video2dataset | 12m25s |
