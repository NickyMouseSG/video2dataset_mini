# To-Do

- [*] Support feather table for sharding meta
- [*] Support Image Downloading
- [*] Support partially downloading video given timestamps
- [*] Integrating all transforms into single processor (comparing to many sub-samplers in original video2dataset)

# Install kn_util
Some basic functions are provided by kn_util (a personal toolkit)
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
