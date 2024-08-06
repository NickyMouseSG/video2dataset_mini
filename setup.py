#!/usr/bin/env python

from distutils.core import setup
import os.path as osp

REQUIREMENT_FELE = "requirement.txt"

def _read_reqs(relpath):
    fullpath = osp.join(osp.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [
            s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))
        ]


REQUIREMENTS = _read_reqs(REQUIREMENT_FELE)

setup(
    name="kv2d",
    version="1.0",
    description="video2dataset_mini",
    author="NickyMouse",
    author_email="jing005@e.ntu.edu.sg",
    packages=["kv2d"],
    entry_points={"console_scripts": ["kv2d = kv2d.main:main"]},
    requires=REQUIREMENTS,
)
