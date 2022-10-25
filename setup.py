#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name="DSRC",
    version="0.1",
    description="Simulation backend for AAS GN&C 2023 paper",
    author="Connor Fuhramn",
    author_email="connorfuhrman@arizona.edu",
    packages=find_packages(),
)
