#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup


setup(
    name='torchfcn',
    version='0.2',
    packages=find_packages(),
    package_data={'torchfcn': ['torchfcn/ext/*']},
    include_package_data=True,
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    license='MIT',
    url='https://github.com/wkentaro/torchfcn',
)
