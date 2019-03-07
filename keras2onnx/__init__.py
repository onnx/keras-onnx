###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
keras-tf-onnx
This package converts keras and tensorflow models into ONNX for use with Windows Machine Learning
"""
__version__ = "1.3.1"
__author__ = "Microsoft Corporation"
__producer__ = "keras2onnx"

__producer_version__ = __version__
__domain__ = "onnx"
__model_version__ = 0

from .main import convert_keras
from .main import convert_keras_tf

from .common import Variable, cvtfunc
from .funcbook import set_converter


def tfname_to_onnx(name): return Variable.tfname_to_onnx(name)


try:
    import tensorflow
except ImportError as e:
    raise AssertionError('Please conda install / pip install tensorflow or tensorflow-gpu before the model conversion.')
