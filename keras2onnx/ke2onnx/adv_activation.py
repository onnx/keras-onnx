###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from ..proto import keras, is_keras_older_than
from ..common.onnx_ops import apply_elu, apply_leaky_relu, apply_prelu, apply_thresholded_relu, apply_clip,\
    OnnxOperatorBuilder
import numpy as np


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
        apply_clip(scope, operator.input_full_names[0], operator.output_full_names[0], container,
                   operator_name=operator.full_name+'_clip', max=op.max_value, min=op.threshold)
    else:
        raise RuntimeError('Unsupported advanced layer found %s' % type(op))


def convert_keras_softmax(scope, operator, container):
    op = operator.raw_operator
    if is_keras_older_than('2.1.3'):
        raise RuntimeError('Unsupported advanced layer found %s' % type(op))
    oopb = OnnxOperatorBuilder(container, scope)
    axis = op.get_config()['axis']
    input_dim = len(op.input_shape)
    if axis == -1:
        oopb.apply_op_with_output('apply_softmax',
                                  operator.input_full_names,
                                  operator.output_full_names,
                                  name=operator.full_name,
                                  axis=-1)
    else:
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
