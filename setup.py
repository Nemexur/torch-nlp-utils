#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()


def setup_package():
    setup(
        name="torch_nlp_utils",
        packages=find_packages(exclude=['tests', 'tests.*']),
        install_requires=required
    )


if __name__ == "__main__":
    setup_package()
