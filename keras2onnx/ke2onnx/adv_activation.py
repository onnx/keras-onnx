# SPDX-License-Identifier: Apache-2.0

from ..proto import keras, is_keras_older_than
from ..common.onnx_ops import apply_elu, apply_leaky_relu, apply_prelu, apply_thresholded_relu, \
    OnnxOperatorBuilder
import numpy as np
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from .._tf_utils import to_onnx_type as _to_onnx_type

activations = keras.layers.advanced_activations


def convert_keras_advanced_activation(scope, operator, container):
    op = operator.raw_operator
    if isinstance(op, activations.LeakyReLU):
        alpha = op.get_config()['alpha']
        apply_leaky_relu(scope, operator.input_full_names[0], operator.output_full_names[0], container,
                         operator_name=operator.full_name, alpha=alpha)
    elif isinstance(op, activations.ELU):
        alpha = op.get_config()['alpha']
        apply_elu(scope, operator.input_full_names[0], operator.output_full_names[0], container,
                  operator_name=operator.full_name, alpha=alpha)
    elif isinstance(op, activations.PReLU):
        weights = op.get_weights()[0]
        apply_prelu(scope, operator.input_full_names[0], operator.output_full_names[0], container,
                    operator_name=operator.full_name, slope=weights)
    elif isinstance(op, activations.ThresholdedReLU):
        alpha = op.get_config()['theta']
        apply_thresholded_relu(scope, operator.input_full_names[0], operator.output_full_names[0], container,
                               operator_name=operator.full_name, alpha=[alpha])
    elif not is_keras_older_than('2.2.0') and isinstance(op, activations.ReLU):
        oopb = OnnxOperatorBuilder(container, scope)
        if abs(op.threshold) > 1e-6:
            raise ValueError("Non-zero ReLU threshold is not supported.")
        else:
            sub_value = operator.input_full_names
        lrelu_value = oopb.apply_leaky_relu(sub_value, name=operator.full_name + '_leaky_relu',
                                            alpha=op.negative_slope.tolist())
        if op.max_value is None:
            oopb.apply_op_with_output("apply_identity",
                                      lrelu_value,
                                      operator.output_full_names,
                                      name=operator.full_name + '_identity')
        else:
            np_type = TENSOR_TYPE_TO_NP_TYPE[operator.inputs[0].type.to_onnx_type().tensor_type.elem_type]
            oopb.apply_op_with_output("apply_min",
                                      lrelu_value +
                                      [('_min', _to_onnx_type(op.input[0].dtype),
                                        np.array(op.max_value, dtype=np_type))],
                                      operator.output_full_names,
                                      name=operator.full_name + '_min')
    else:
        raise RuntimeError('Unsupported advanced layer found %s' % type(op))


def convert_keras_softmax(scope, operator, container):
    op = operator.raw_operator
    if is_keras_older_than('2.1.3'):
        raise RuntimeError('Unsupported advanced layer found %s' % type(op))
    oopb = OnnxOperatorBuilder(container, scope)
    axis = op.get_config()['axis']
    input_dim = len(op.input_shape)
    if axis == -1 or axis == input_dim - 1:
        oopb.apply_op_with_output('apply_softmax',
                                  operator.input_full_names,
                                  operator.output_full_names,
                                  name=operator.full_name,
                                  axis=-1)
    else:
        if axis < 0:
            axis += input_dim
        perm_1 = list(range(0, axis)) + list(range(axis + 1, input_dim)) + [axis]
        inverse_perm = np.arange(len(perm_1))[np.argsort(perm_1)]
        transpose_1 = oopb.apply_transpose(operator.input_full_names,
                                           name=operator.full_name + '_transpose_0',
                                           perm=perm_1)
        softmax_0 = oopb.apply_softmax(transpose_1,
                                       name=operator.full_name + '_softmax_0',
                                       axis=-1)
        oopb.apply_op_with_output('apply_transpose',
                                  softmax_0,
                                  operator.output_full_names,
                                  name=operator.full_name + '_transpose_1',
                                  perm=inverse_perm)
