# -*- coding: utf-8 -*-
###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################


from distutils.core import setup
from setuptools import find_packages
import os
from os import path


def find_embedded_package(folder):
    packages = find_packages(where=folder)
    merged = [p_ if p_.startswith(folder) else folder + '.' + p_ for p_ in packages]
    return [p_[2:].replace('/', '.') for p_ in merged]


this = os.path.dirname(__file__)
with open(os.path.join(this, "requirements.txt"), "r") as f:
    requirements = [_ for _ in [_.strip("\r\n ")
                                for _ in f.readlines()] if _ is not None]

packages = find_packages()
assert packages
root_package = packages[0]
packages += find_embedded_package('./keras2onnx/ktf2onnx')

# read version from the package file.
version_str = '0.1.0.0000'
with (open(os.path.join(this, '{}/__init__.py'.format(root_package)), "r")) as f:
    line = [_ for _ in [_.strip("\r\n ")
                        for _ in f.readlines()] if _.startswith("__version__")]
    if len(line) > 0:
        version_str = line[0].split('=')[1].strip('" ')


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    start_pos = long_description.find('# Introduction')
    if start_pos >= 0:
        long_description = long_description[start_pos:]

setup(
    name=root_package,
    version=version_str,
    description="Converts Machine Learning models to ONNX for use in Windows ML",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT License',
    author='Microsoft Corporation',
    author_email='winmlcvt@microsoft.com',
    url='https://github.com/onnx/keras-onnx',
    packages=packages,
    include_package_data=True,
    install_requires=requirements,
    tests_require=['pytest', 'pytest-cov'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License']
)
