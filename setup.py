#!/usr/bin/env python

from distutils.core import setup
import os.path as osp


setup(
    name="kv2d",
    version="1.0",
    description="video2dataset_mini",
    author="NickyMouse",
    author_email="jing005@e.ntu.edu.sg",
    packages=["kv2d"],
    entry_points={"console_scripts": ["kv2d = kn_v2d.main:main"]},
)
