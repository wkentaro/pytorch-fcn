#!/usr/bin/env python

from __future__ import print_function

from distutils.version import LooseVersion
import sys

from setuptools import find_packages
from setuptools import setup


__version__ = '1.6.1'


try:
    import torch  # NOQA
    if LooseVersion(torch.__version__) < LooseVersion('0.2.0'):
        raise ImportError
except ImportError:
    print('Please install pytorch>=0.2.0. (See http://pytorch.org)',
          file=sys.stderr)
    sys.exit(1)


setup(
    name='torchfcn',
    version=__version__,
    packages=find_packages(),
    install_requires=[r.strip() for r in open('requirements.txt')],
    description='PyTorch Implementation of Fully Convolutional Networks.',
    package_data={'torchfcn': ['ext/*']},
    include_package_data=True,
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    license='MIT',
    url='https://github.com/wkentaro/pytorch-fcn',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
        'Topic :: Internet :: WWW/HTTP',
    ],
)
