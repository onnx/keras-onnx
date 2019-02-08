# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from distutils.core import setup
from setuptools import find_packages
import os
from os import path

this = os.path.dirname(__file__)

with open(os.path.join(this, "requirements.txt"), "r") as f:
    requirements = [_ for _ in [_.strip("\r\n ")
                                for _ in f.readlines()] if _ is not None]

packages = find_packages()
assert packages
pkgname = packages[0]

# read version from the package file.
version_str = '0.1.0.0000'
with (open(os.path.join(this, '{}/__init__.py'.format(pkgname)), "r")) as f:
    line = [_ for _ in [_.strip("\r\n ")
                        for _ in f.readlines()] if _.startswith("__version__")]
    if len(line) > 0:
        version_str = line[0].split('=')[1].strip('" ')


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    start_pos = long_description.find('## Introduction')
    if start_pos >= 0:
        long_description = long_description[start_pos:]

setup(
    name=pkgname,
    version=version_str,
    description="Converts Machine Learning models to ONNX for use in Windows ML",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT License',
    author='Microsoft Corporation',
    author_email='winmlcvt@microsoft.com',
    url='https://microsoft.com',
    packages=packages,
    include_package_data=True,
    install_requires=requirements,
    tests_require=['pytest', 'pytest-cov'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Operating System :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License'],
)
