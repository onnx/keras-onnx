###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

from ..proto import keras
from ..proto.tfcompat import tensorflow as tf
from ..common import name_func
from ..common.onnx_ops import apply_relu_6, apply_softmax, apply_shape, apply_gather, \
    apply_squeeze, apply_transpose, OnnxOperatorBuilder
from .activation import activation_map
from onnx import numpy_helper
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
import numpy as np
activation_get = keras.activations.get


def get_permutation_config(n_dims):
    input_perm_axes = [0, n_dims + 1] + list(range(1, n_dims + 1))
    output_perm_axes = [0] + list(range(2, n_dims + 2)) + [1]
    return input_perm_axes, output_perm_axes


def activation_process(scope, operator, container, biased_tensor_name):
    # Create an activation function node and apply activation function to the intermediate tensor
    apply_activation_function = activation_map[operator.raw_operator.activation]
    if operator.raw_operator.activation in [activation_get('softmax'), keras.activations.softmax]:
        apply_softmax(scope, biased_tensor_name, operator.outputs[0].full_name, container, axis=-1)
    elif operator.raw_operator.activation in [tf.nn.relu6]:
        np_type = TENSOR_TYPE_TO_NP_TYPE[operator.inputs[0].type.to_onnx_type().tensor_type.elem_type]
        zero_value = np.zeros(shape=(1,), dtype=np_type)
        apply_relu_6(scope, biased_tensor_name, operator.outputs[0].full_name, container,
                     zero_value=zero_value)
    else:
        apply_activation_function(scope, biased_tensor_name, operator.outputs[0].full_name, container)


def reverse_sequence(scope, container, input_name, output_name, name, axes):
    oopb = OnnxOperatorBuilder(container, scope)
    rv2_in_names = [input_name]
    apply_shape(scope, input_name, input_name + '_shape', container)
    rv2_node_name = name
    inputs = rv2_in_names

    axis = axes[0]
    batch_axis = 1 if axis != 1 else 0

    const_batch = numpy_helper.from_array(np.array([batch_axis], dtype=np.int64), rv2_node_name + '_const_batch')
    container.add_initializer_from_tensor(const_batch)
    const_axis = numpy_helper.from_array(np.array([axis], dtype=np.int64), rv2_node_name + '_const_axis')
    container.add_initializer_from_tensor(const_axis)

    apply_gather(scope, [input_name + '_shape', const_batch.name], rv2_node_name + '_gather_batch', container)
    apply_gather(scope, [input_name + '_shape', const_axis.name], rv2_node_name + '_gather_axis', container)
    seq_array = oopb.add_node('Expand', [rv2_node_name + '_gather_axis', rv2_node_name + '_gather_batch'],
                              rv2_node_name + '_expand')
    inputs.append(seq_array)

    res_seq_node = oopb.add_node('ReverseSequence', inputs, name=rv2_node_name + '_rev_seq', batch_axis=batch_axis,
                                 time_axis=axis, op_version=10)

    oopb.apply_op_with_output('apply_identity', [res_seq_node], [output_name],
                              name=rv2_node_name + '_Identity')


def reverse_output_adjustment(scope, operator, container, y, h, output_seq, time_major, direction):
    _name = name_func(scope, operator)

    output_name = operator.outputs[0].full_name

    # Create output-adjusting operators
    if output_seq:
        # Squeeze the num_direction dim as we know its size is 1 for
        # lstm/gru(forward/reverse).
        is_reverse = True if direction == 'reverse' else False
        lstm_out = output_name if time_major else _name('y_squeezed')
        squeeze_out = lstm_out if not is_reverse else _name('y_squeezed')
        apply_squeeze(scope, y, squeeze_out, container, axes=[1])

        if time_major:
            if is_reverse:
                reverse_sequence(scope, container, lstm_out, output_name, name=_name('reverse_seq'), axes=[0])

        else:
            # Onnx LSTM/GRU produces time major output. Add a transpose operator to
            # make it batch_major, if the keras op was not time_major.
            # This transforms [ S, B, I] -> [ B, S, I ] where B is
            # batch_size and S is seq_len.
            perm = [1, 0, 2]
            transpose_out = output_name if not is_reverse else _name('transpose')
            apply_transpose(scope, squeeze_out, transpose_out, container, perm=perm)
            if is_reverse:
                reverse_sequence(scope, container, transpose_out, output_name, name=_name('reverse_seq'), axes=[1])

    else:
        apply_squeeze(scope, h, output_name, container, axes=[0])
