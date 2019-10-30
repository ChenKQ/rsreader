#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    # Metadata
    name='rsreader',
    version='20191030',
    author='chenkq',
    url='https://github.com/chenkq/rsreader',
    description='A simple reader for remote sensing',
    long_description='',
    license='',

    # Package info
    # packages=["rsreader","rsreader/utility","rsreader/lvreader", "rsreader/store", "rsreader/netreader", \
    #           "rsreader/netreader/torchreader", "rsreader/netreader/mxreader"],
    packages=find_packages(exclude=('test')),
    # package_dir = {'':'rsreader'},

    zip_safe=True,
    include_package_data=True,
    # install_requires=requirements,
)