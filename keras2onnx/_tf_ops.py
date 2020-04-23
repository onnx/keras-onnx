# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

from .funcbook import converter_func
from ._tf_utils import (tf_attrs_to_onnx as _to_onnx_attrs,
                        cal_tensor_shape as _cal_tensor_shape,
                        to_onnx_type as _to_onnx_type)


def _random_converter(scope, operator, container):
    tf_op = operator.raw_operator
    op_type = tf_op.type
    if op_type == 'RandomStandardNormal':
        op_type = 'RandomNormal'
    inputs = [var_.full_name for var_ in operator.inputs]

    attrs = {}
    shape = _cal_tensor_shape(tf_op.inputs[0])
    attrs['shape'] = shape
    del inputs[0]

    if 'seed' in _to_onnx_attrs(tf_op):
        attrs['seed'] = float(tf_op.get_attr('seed'))

    container.add_node(op_type,
                       inputs,
                       [var_.full_name for var_ in operator.outputs],
                       name=operator.raw_operator.name,
                       op_version=1,
                       **attrs
                       )


@converter_func(
    "RandomNormal",
    'RandomStandardNormal',
    "RandomUniform")
def convert_tf_random_standard_normal(scope, operator, container):
    _random_converter(scope, operator, container)


def pass_thru_converter(scope, operator, container):
    tf_op = operator.raw_operator
    attrs = _to_onnx_attrs(tf_op)

    container.add_node(operator.type,
                       operator.input_full_names,
                       operator.output_full_names,
                       name=operator.full_name,
                       **attrs)
