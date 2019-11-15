###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from .proto.tfcompat import tensorflow as tf, normalize_tensor_shape
from onnxconverter_common import Int32TensorType, Int64TensorType, FloatTensorType, DoubleTensorType, BooleanTensorType


def _infer_variable_type(tensor, opset):
    tensor_shape = []
    if tensor.shape not in (tf.TensorShape(None), tf.TensorShape([])):
        if opset > 8:
            tensor_shape = normalize_tensor_shape(tensor.shape)
        else:  # most inference engine has problem with unset dim param if they released around opset 8 publish
            tensor_shape = ['None' if d is None else d for d in normalize_tensor_shape(tensor.shape)]

    # Determine the tensor's element type
    tensor_type = tensor.dtype
    if tensor.dtype == 'resource':
        node_attr = tensor.op.node_def.attr
        tensor_type = node_attr['dtype'].type
        tensor_shape = ['None' if d.size is None else d.size for d in node_attr['shape'].shape.dim]
    if tensor_type in [tf.int8, tf.int16, tf.int32]:
        return Int32TensorType(shape=tensor_shape)
    elif tensor_type == tf.int64:
        return Int64TensorType(shape=tensor_shape)
    elif tensor_type in [tf.float16, tf.float32]:
        return FloatTensorType(shape=tensor_shape)
    elif tensor_type == tf.float64:
        return DoubleTensorType(shape=tensor_shape)
    elif tensor_type == tf.bool:
        return BooleanTensorType(shape=tensor_shape)
    else:
        raise ValueError(
            "Unable to find out a correct type for tensor type = {} of {}".format(tensor_type, tensor.name))
