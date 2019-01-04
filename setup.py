#!/usr/bin/env python

import distutils.spawn
import shlex
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup

import github2pypi


version = '1.9.3'


if sys.argv[1] == 'release':
    if not distutils.spawn.find_executable('twine'):
        print(
            'Please install twine:\n\n\tpip install twine\n',
            file=sys.stderr,
        )
        sys.exit(1)

    commands = [
        'git pull origin master',
        'git tag v{:s}'.format(version),
        'git push origin master --tag',
        'python setup.py sdist',
        'twine upload dist/torchfcn-{:s}.tar.gz'.format(version),
    ]
    for cmd in commands:
        print('+ {}'.format(cmd))
        subprocess.check_call(shlex.split(cmd))
    sys.exit(0)


with open('README.md') as f:
    long_description = github2pypi.replace_url(
        slug='wkentaro/pytorch-fcn', content=f.read()
    )


setup(
    name='torchfcn',
    version=version,
    packages=find_packages(exclude=['github2pypi']),
    install_requires=[r.strip() for r in open('requirements.txt')],
    description='PyTorch Implementation of Fully Convolutional Networks.',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
