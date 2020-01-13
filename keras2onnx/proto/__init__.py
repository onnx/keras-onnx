###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import onnx
import tensorflow
from distutils.version import StrictVersion

# Rather than using ONNX protobuf definition throughout our codebase, we import ONNX protobuf definition here so that
# we can conduct quick fixes by overwriting ONNX functions without changing any lines elsewhere.
from onnx import onnx_pb as onnx_proto
from onnx import helper


def get_opset_number_from_onnx():
    return onnx.defs.onnx_opset_version()


def _check_onnx_version():
    import pkg_resources
    min_required_version = pkg_resources.parse_version('1.0.1')
    current_version = pkg_resources.get_distribution('onnx').parsed_version
    assert current_version >= min_required_version, 'Keras2ONNX requires ONNX version 1.0.1 or a newer one'


_check_onnx_version()
is_tf_keras = False
if os.environ.get('TF_KERAS', '0') != '0':
    is_tf_keras = True


if is_tf_keras:
    from tensorflow.python import keras
else:
    try:
        import keras
    except ImportError:
        is_tf_keras = True
        from tensorflow.python import keras


def is_keras_older_than(version_str):
    return StrictVersion(keras.__version__.split('-')[0]) < StrictVersion(version_str)


def is_keras_later_than(version_str):
    return StrictVersion(keras.__version__.split('-')[0]) > StrictVersion(version_str)


def is_tensorflow_older_than(version_str):
    return StrictVersion(tensorflow.__version__.split('-')[0]) < StrictVersion(version_str)

def is_tensorflow_later_than(version_str):
    return StrictVersion(tensorflow.__version__.split('-')[0]) > StrictVersion(version_str)
