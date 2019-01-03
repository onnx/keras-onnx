###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
keras-tf-onnx
This package converts keras and tensorflow models into ONNX for use with Windows Machine Learning
"""
__version__ = "0.0.1"
__author__ = "Microsoft Corporation"
__producer__ = "ketone"

__producer_version__ = __version__
__domain__ = "onnxml"
__model_version__ = 0

from .main import convert_keras
from .main import convert_keras_tf

from .common import Variable, cvtfunc
from .funcbook import set_converter


def tfname_to_onnx(name): return Variable.tfname_to_onnx(name)
