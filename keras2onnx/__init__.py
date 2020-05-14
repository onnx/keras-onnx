###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
"""
keras2onnx
This package converts keras models into ONNX for use with any inference engine supporting ONNX
"""
__version__ = "1.6.9"
__author__ = "Microsoft Corporation"
__producer__ = "keras2onnx"

__producer_version__ = __version__
__domain__ = "onnx"
__model_version__ = 0

try:
    import sys
    import os.path
    from os.path import dirname, abspath
    import tensorflow
    from distutils.version import StrictVersion

    if StrictVersion(tensorflow.__version__.split('-')[0]) >= StrictVersion('2.0.0'):
        tensorflow.compat.v1.disable_tensor_equality()
except ImportError:
    raise AssertionError('Please conda install / pip install tensorflow or tensorflow-gpu before the model conversion.')

from .proto import save_model
from .common import Variable, cvtfunc, set_logger_level
from .funcbook import set_converter, set_converters

from .main import convert_keras
from .main import export_tf_frozen_graph
from .main import build_io_names_tf2onnx


def tfname_to_onnx(name): return Variable.tfname_to_onnx(name)
