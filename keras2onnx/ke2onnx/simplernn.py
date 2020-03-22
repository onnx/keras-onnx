###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numpy as np
from ..proto import onnx_proto
from ..common.onnx_ops import apply_reshape, apply_transpose, apply_cast, OnnxOperatorBuilder
from .common import extract_recurrent_activation

TensorProto = onnx_proto.TensorProto


def extract_input_shape(op):
    """Returns the input shape for a RNN class.
    """
    input_shape = op.get_input_shape_at(0)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    return input_shape

def extract_params(op, hidden_size):
    """Returns a tuple of the SimpleRNN parameters, and converts them into the format for ONNX.
    """
    params = op.get_weights()

    W = params[0].T
    R = params[1].T

    B = None
    if op.use_bias:
        B = np.zeros((2, hidden_size), dtype=np.float32)
        B[0] = params[2]

    return W, R, B

def build_sequence_lengths(scope, operator, container):
    """Uses the masking layer to calculate the sequence lengths.
    """
    input_mask_name = operator.input_masks[0].full_name
    mask_cast = scope.get_unique_operator_name(operator.full_name + '_mask_cast')
    sequence_lengths = scope.get_unique_operator_name(operator.full_name + '_seq_lens')

    apply_cast(scope, input_mask_name, mask_cast, container, to=TensorProto.INT32)
    container.add_node('ReduceSum', mask_cast, sequence_lengths, keepdims=False, axes=[-1])
    return sequence_lengths

def convert_keras_simple_rnn(scope, operator, container):
    op = operator.raw_operator
    hidden_size = op.units
    _, seq_length, input_size = extract_input_shape(op)
    output_seq = op.return_sequences
    output_state = op.return_state
    reverse_input = op.go_backwards

    W, R, B = extract_params(op, hidden_size)

    attrs = {}
    rnn_input_names = []
    rnn_output_names = []

    rnn_x_name = scope.get_unique_variable_name('rnn_x')
    apply_transpose(scope, operator.inputs[0].full_name, rnn_x_name, container, perm=[1, 0, 2])
    rnn_input_names.append(rnn_x_name)

    tensor_w_name = scope.get_unique_variable_name('tensor_w')
    container.add_initializer(tensor_w_name, onnx_proto.TensorProto.FLOAT, [1, hidden_size, input_size], W.flatten())
    rnn_input_names.append(tensor_w_name)

    tensor_r_name = scope.get_unique_variable_name('tensor_r')
    container.add_initializer(tensor_r_name, onnx_proto.TensorProto.FLOAT, [1, hidden_size, hidden_size], R.flatten())
    rnn_input_names.append(tensor_r_name)

    if op.use_bias:
        tensor_b_name = scope.get_unique_variable_name('tensor_b')
        container.add_initializer(tensor_b_name, onnx_proto.TensorProto.FLOAT, [1, 2 * hidden_size], B.flatten())
        rnn_input_names.append(tensor_b_name)
    else:
        rnn_input_names.append('')

    # sequence_lens is not able to be converted from input_length
    uses_masking_layer = len(operator.input_masks) == 1
    if uses_masking_layer:
        # Mask using sequence_lens input
        sequence_lengths = build_sequence_lengths(scope, operator, container)
        rnn_input_names.append(sequence_lengths)
    else:
        rnn_input_names.append('')
    # inital_h: none
    if len(operator.inputs) == 1:
        rnn_input_names.append('')
    else:
        # Add a reshape after initial_h, 2d -> 3d
        input_reshape_name = scope.get_unique_variable_name('input_reshape')
        apply_reshape(scope, operator.inputs[1].full_name, input_reshape_name, container,
                      desired_shape=[1, -1, hidden_size])
        rnn_input_names.append(input_reshape_name)

    if hasattr(op, 'activation'):
        activation_type, alpha, beta = extract_recurrent_activation(op.activation)
        attrs['activations'] = [activation_type.encode('utf-8')]
        if alpha is not None:
            attrs['activation_alpha'] = [alpha]
        if beta is not None:
            attrs['activation_beta'] = [beta]

    attrs['direction'] = 'reverse' if reverse_input else 'forward'
    attrs['hidden_size'] = hidden_size

    # We use the collected information to build ONNX's RNN. ONNX RNN's outputs will be saved onto two intermediate
    # tensors and we will adjust them subsequently to mimic Keras output format.
    rnn_y_name = scope.get_unique_variable_name('rnn_y')
    rnn_h_name = scope.get_unique_variable_name('rnn_h')
    rnn_output_names.append(rnn_y_name)
    rnn_output_names.append(rnn_h_name)
    oopb = OnnxOperatorBuilder(container, scope)

    oopb.apply_op_with_output('apply_rnn',
                              rnn_input_names,
                              rnn_output_names,
                              name=operator.raw_operator.name,
                              output_seq=output_seq,
                              **attrs)

    # Create operators to adjust ONNX output to meet Keras format
    if output_seq:
        permuted_rnn_y_name = scope.get_unique_variable_name('rnn_y_permuted')
        perm = [1, 0, 2] if container.target_opset <= 5 else [2, 0, 1, 3]
        apply_transpose(scope, rnn_y_name, permuted_rnn_y_name, container, perm=perm)
        apply_reshape(scope, permuted_rnn_y_name, operator.outputs[0].full_name, container,
                      desired_shape=[-1, seq_length, hidden_size])
    else:
        # Here we ingore ONNX RNN's first output because it's useless.
        apply_reshape(scope, rnn_h_name, operator.outputs[0].full_name, container, desired_shape=[-1, hidden_size])

    if output_state:
        apply_reshape(scope, rnn_h_name, operator.outputs[1].full_name, container, desired_shape=[-1, hidden_size])
