#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    # Metadata
    name='rsreader',
    version='20181204',
    author='chenkq',
    url='https://github.com/chenkq/rsreader',
    description='A simple reader for remote sensing',
    long_description='',
    license='',

    # Package info
    packages=find_packages('rsreader'),
    package_dir = {'':'rsreader'},

    zip_safe=True,
    include_package_data=True,
    # install_requires=requirements,
)