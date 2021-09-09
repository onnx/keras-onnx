# SPDX-License-Identifier: Apache-2.0

"""
keras2onnx
This package converts keras models into ONNX for use with any inference engine supporting ONNX
"""
__version__ = "1.9.0"
__author__ = "Microsoft Corporation"
__producer__ = "keras2onnx"

__producer_version__ = __version__
__domain__ = "onnxmltools"
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

from .proto import save_model, is_tensorflow_later_than
from .common import Variable, cvtfunc, set_logger_level
from .common.utils import k2o_logger
from .funcbook import set_converter, set_converters

if is_tensorflow_later_than("2.3"):
    start_red = "\033[91m"
    end_color = "\033[00m"
    k2o_logger().error(
        start_red + "\n**** keras2onnx does not support tensorflow version > 2.3. "
        "Please see https://github.com/onnx/keras-onnx/issues/737 ****\n" + end_color)

from .main import convert_keras
from .main import export_tf_frozen_graph
from .main import build_io_names_tf2onnx


def tfname_to_onnx(name): return Variable.tfname_to_onnx(name)
