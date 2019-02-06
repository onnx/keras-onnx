###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from .common.onnx_ops import apply_identity
from .funcbook import set_converter


def default_convert(scope, operator, container):
    apply_identity(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container)


# def reshape_shape(operator):
#     oop = operator.raw_operator[0]
#     shape_op = oop.inputs[1].op  # type: tf.Operation
#     assert shape_op.type == 'Pack'
#
#     shape = [0, ]
#     for in_ in shape_op.inputs:
#         op_node = in_.op
#         if op_node.type != 'Const':
#             continue
#         is_raw, n = get_tf_tensor_data(op_node.get_attr('value'))
#         assert is_raw is False
#         shape.append(n[0])
#
#     var_type = operator.inputs[0].type
#     var_type.shape = shape
#     operator.outputs[0].type = var_type
#
#
# @cvtfunc(shape_infer=reshape_shape, pattern=r'(^.*/reshape_\d+/)')
# def reshape_convert(scope, operator, container):
#     apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
#                   desired_shape=operator.outputs[0].type.shape)
#
#
# def flatten_shape(operator):
#     oop = operator.raw_operator[0]
#     shape = oop.inputs[0].shape  # type: tf.Operation
#     accum = 1
#     for i_ in shape[1:]:
#         accum *= i_.value
#     operator.outputs[0].type.shape = [DEFAULT_BATCH_SIZE, accum]
#
#
# @cvtfunc(shape_infer=flatten_shape, pattern=r'(^.*/flatten_\d+/)')
# def flatten_convert(scope, operator, container):
#     apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
#                   desired_shape=[0, operator.outputs[0].type.shape[1]])
#
#
# set_converter('Flatten', flatten_convert)
# set_converter('Reshape', reshape_convert)
set_converter('identity', default_convert)
