###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numbers
import numpy as np
from onnx import numpy_helper, mapping
from .common.onnx_ops import apply_identity, apply_reshape, OnnxOperatorBuilder
from .funcbook import converter_func, set_converters
from .proto import onnx_proto
from .proto.tfcompat import tensorflow


class TYPES:
    # tf-node types:
    Identity = 'Identity'
    Const = 'Const'
    Any = 'Any'
    All = 'All'
    Cast = 'Cast'

    # converter internal types:
    TD_Reshape = '_reshape_timedistributed'


def _cal_tensor_value(tensor):  # type: (tensorflow.Tensor)->np.ndarray
    node = tensor.op
    assert node.type in ["Const", "ConstV2"], "{} has to be a constant".format(node.name)
    make_ndarray = tensorflow.make_ndarray
    np_arr = make_ndarray(node.get_attr("value"))
    return np_arr


def _to_onnx_type(dt_type):
    # TensorFlow data types intergrate seamlessly with numpy
    return mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dt_type.as_numpy_dtype)]


def default_convert(scope, operator, container):
    apply_identity(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container)


@converter_func(TYPES.Identity)
def convert_tf_identity(scope, operator, container):
    default_convert(scope, operator, container)


@converter_func(TYPES.Const)
def convert_tf_const(scope, operator, container):
    node = operator.raw_operator
    np_arr = _cal_tensor_value(node.outputs[0])
    onnx_tensor = numpy_helper.from_array(np_arr, node.outputs[0].name)
    container.add_initializer_from_tensor(onnx_tensor)


@converter_func(TYPES.TD_Reshape)
def convert_reshape_timedistributed(scope, operator, container):
    target_shape = operator.get_attr('target_shape')
    apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
                  operator_name=operator.full_name, desired_shape=target_shape)


@converter_func(TYPES.All, TYPES.Any)
def convert_tf_any_all(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    node = operator.raw_operator
    axis = _cal_tensor_value(node.inputs[1]).tolist()
    axis = [axis] if np.isscalar(axis) else axis

    # It is fine to have nagative reduce_dim.
    cast_op = oopb.apply_cast(operator.input_full_names[0],
                              to=onnx_proto.TensorProto.FLOAT,
                              name=operator.full_name + '_cast')
    keepdims = node.get_attr("keep_dims")
    op_type = "ReduceMin" if node.type == "All" else "ReduceSum"
    reduce_op = oopb.add_node(op_type, cast_op,
                              axes=axis,
                              keepdims=keepdims,
                              name=operator.full_name + '_reduce')

    oopb.add_node_with_output("Greater",
                              [reduce_op, np.array(0, dtype=np.float32)],
                              operator.output_full_names,
                              name=operator.full_name,
                              op_version=9)


@converter_func(TYPES.Cast)
def convert_tf_case(scope, operator, container):
    node = operator.raw_operator
    to = _to_onnx_type(node.get_attr("DstT"))
    oopb = OnnxOperatorBuilder(container, scope)
    oopb.apply_op_with_output("apply_cast",
                              operator.input_full_names,
                              operator.output_full_names,
                              name=operator.full_name,
                              to=to)


direct_ops = {"Abs": ("apply_abs",),
              "Acos": 7,
              "Acosh": 9,
              "Add": ("apply_add",),
              "Asin": 7,
              "Asinh": 9,
              "Atan": 7,
              "Atanh": 9,
              "Ceil": ("apply_ceil",),
              "Cos": 7,
              "Cosh": 9,
              "Div": ("apply_div",),
              "Elu": ("apply_elu",),
              "Exp": ("apply_exp",),
              "Floor": ("apply_floor",),
              "Log": ("apply_log",),
              "Mul": ("apply_mul",),
              "Neg": ("apply_neg",),
              "Pow": ("apply_pow",),
              "Reciprocal": ("apply_reciprocal",),
              "Relu": ("apply_relu",),
              "Round": 11,
              "Sigmoid": ("apply_sigmoid",),
              "Sin": 7,
              "Sinh": 9,
              "Softplus": 1,
              "Softsign": 1,
              "Sqrt": ("apply_sqrt",),
              "Sub": ("apply_sub",),
              "Tan": 7,
              "Tanh": ("apply_tanh",)
              }


def tf_op_convert(scope, operator, container):
    oopb = OnnxOperatorBuilder(container, scope)
    type = operator.raw_operator.type
    item = direct_ops[type]
    assert item is not None, "Can't find the tf op item."
    if isinstance(item, numbers.Integral):
        oopb.add_node_with_output(type,
                                  [var_.full_name for var_ in operator.inputs],
                                  [var_.full_name for var_ in operator.outputs],
                                  name=operator.raw_operator.name,
                                  op_version=item
                                  )
    else:
        apply_func_name = item[0]
        oopb.apply_op_with_output(apply_func_name,
                                  [var_.full_name for var_ in operator.inputs],
                                  [var_.full_name for var_ in operator.outputs],
                                  name=operator.raw_operator.name,
                                  )


set_converters({k: tf_op_convert for k in direct_ops.keys()})
