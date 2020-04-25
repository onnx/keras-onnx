# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from onnxconverter_common.oopb import OnnxOperatorBuilder
from .funcbook import converter_func
from ._tf_utils import tf_attrs_to_onnx as _to_onnx_attrs


def _random_converter(scope, operator, container):
    tf_op = operator.raw_operator
    op_type = tf_op.type
    if op_type == 'RandomStandardNormal':
        op_type = 'RandomNormalLike'
    else:
        op_type = op_type + 'Like'
    inputs = [var_.full_name for var_ in operator.inputs]

    attrs = {}
    if 'seed2' in _to_onnx_attrs(tf_op):
        attrs['seed'] = float(tf_op.get_attr('seed2'))
    with OnnxOperatorBuilder(container, scope).as_default(tf_op.name) as oopb:
        cast_n = oopb.cast(inputs, to=oopb.int64)
        const_op = oopb.add_node('ConstantOfShape', cast_n, op_version=9)
        oopb.add_node(op_type, const_op,
                      outputs=[var_.full_name for var_ in operator.outputs],
                      op_version=1, **attrs)


@converter_func(
    "RandomNormal",
    'RandomStandardNormal',
    "RandomUniform")
def convert_tf_random_standard_normal(scope, operator, container):
    _random_converter(scope, operator, container)


def pass_thru_converter(scope, operator, container):
    """
    This the converter to copy the original graph node with its def into a ONNX node format.
    """
    tf_op = operator.raw_operator
    attrs = _to_onnx_attrs(tf_op)

    container.add_node(operator.type,
                       operator.input_full_names,
                       operator.output_full_names,
                       name=operator.full_name,
                       op_domain='ai.onnx.contrib',
                       op_version=1,
                       **attrs)
