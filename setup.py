#!/usr/bin/env python

import shlex
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup


__version__ = '1.9.0'


if sys.argv[-1] == 'release':
    commands = [
        'python setup.py sdist',
        'twine upload dist/torchfcn-{0}.tar.gz'.format(__version__),
        'git tag v{0}'.format(__version__),
        'git push origin master --tags',
    ]
    for cmd in commands:
        subprocess.call(shlex.split(cmd))
    sys.exit(0)


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
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
