###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numpy as np
from ..proto import onnx_proto, keras
from ..common import name_func
from ..common.onnx_ops import apply_reshape, apply_transpose, apply_cast, OnnxOperatorBuilder

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


def extract_recurrent_activation(activation):
    activations = keras.activations
    alpha = None
    beta = None
    if activation == activations.sigmoid:
        onnx_op_type = 'Sigmoid'
    elif activation == activations.hard_sigmoid:
        onnx_op_type = 'HardSigmoid'
        alpha = 0.2
        beta = 0.5
    elif activation == activations.tanh:
        onnx_op_type = 'Tanh'
    elif activation == activations.relu:
        onnx_op_type = 'Relu'
    elif activation == activations.linear:
        onnx_op_type = 'Affine'
        alpha = 1.0
    else:
        raise NotImplementedError('The activation %s not supported' % activation)

    return onnx_op_type, alpha, beta


def extract_activations(fields):
    """Returns a dictionary with the appropriate activations set
    """
    activation_types = []
    alphas = []
    betas = []
    activations = [extract_recurrent_activation(f) for f in fields]
    for (activation_type, alpha, beta) in activations:
        activation_types.append(activation_type.encode('utf-8'))
        if alpha is not None:
            alphas.append(alpha)
        if beta is not None:
            betas.append(beta)

    attrs = {}
    attrs['activations'] = activation_types
    if alphas:
        attrs['activation_alpha'] = alphas
    if betas:
        attrs['activation_beta'] = betas
    return attrs

def build_parameters(scope, operator, container, bidirectional=False):
    """Returns the parameter initialization values after extracting them from the RNN layer.
    """
    op = operator.raw_operator
    _, seq_length, input_size = extract_input_shape(op)

    _name = name_func(scope, operator)

    tensor_w = _name('W')
    tensor_r = _name('R')
    tensor_b = ''

    if bidirectional:
        forward_layer = op.forward_layer
        backward_layer = op.backward_layer
        hidden_size = forward_layer.units

        W, R, B = extract_params(forward_layer, hidden_size)
        W_back, R_back, B_back = extract_params(backward_layer, hidden_size)

        W = np.concatenate([W, W_back])
        W_shape = [2, hidden_size, input_size]

        R = np.concatenate([R, R_back])
        R_shape = [2, hidden_size, hidden_size]

        if (B is None and B_back is not None) or (B is not None and B_back is None):
            raise ValueError('Bidirectional bias must be enabled (or disabled) for both forward '
                             'and backward layers.')

        if B is not None:
            B = np.concatenate([B, B_back])
            B_shape = [2, 2 * hidden_size]

    else:
        hidden_size = op.units

        W, R, B = extract_params(op, hidden_size)
        W_shape = [1, hidden_size, input_size]
        R_shape = [1, hidden_size, hidden_size]

        if B is not None:
            B_shape = [1, 2 * hidden_size]

    # Create initializers
    container.add_initializer(tensor_w, TensorProto.FLOAT, W_shape, W.flatten())
    container.add_initializer(tensor_r, TensorProto.FLOAT, R_shape, R.flatten())

    if B is not None:
        tensor_b = _name('B')
        container.add_initializer(tensor_b, TensorProto.FLOAT, B_shape, B.flatten())

    return tensor_w, tensor_r, tensor_b


def build_sequence_lengths(scope, operator, container):
    """Uses the masking layer to calculate the sequence lengths. If there is no
    masking layer, then it returns an empty input for the sequence lengths.
    """
    # Masking input must be present
    if len(operator.input_masks) != 1:
        return ''

    input_mask_name = operator.input_masks[0].full_name
    mask_cast = scope.get_unique_operator_name(operator.full_name + '_mask_cast')
    sequence_lengths = scope.get_unique_operator_name(operator.full_name + '_seq_lens')

    apply_cast(scope, input_mask_name, mask_cast, container, to=TensorProto.INT32)
    container.add_node('ReduceSum', mask_cast, sequence_lengths, keepdims=False, axes=[-1])
    return sequence_lengths


def build_initial_states(scope, operator, container, bidirectional=False):
    """Reshapes the initial input states. If there are no states present as inputs, then
    it returns an empty input for the initial hidden states.
    """
    # Initial hidden states
    if len(operator.inputs) == 1:
        return ''

    # Add a reshape after initial_h, 2d -> 3d
    hidden_size = operator.raw_operator.units
    input_h = operator.inputs[1].full_name
    initial_h = scope.get_unique_variable_name(operator.full_name + '_initial_h')
    desired_shape = [2, -1, hidden_size] if bidirectional else [1, -1, hidden_size]
    apply_reshape(scope, input_h, initial_h, container, desired_shape=desired_shape)
    return initial_h


def build_attributes(scope, operator, container, bidirectional=False):
    """Returns a dictionary of attributes for the LSTM layer.
    """
    op = operator.raw_operator

    attrs = {}

    if bidirectional:
        forward_layer = op.forward_layer
        backward_layer = op.backward_layer

        attrs['direction'] = 'bidirectional'
        attrs['hidden_size'] = forward_layer.units

        activations = []
        if hasattr(forward_layer, 'activation'):
            activations.append(forward_layer.activation)

        if hasattr(backward_layer, 'activation'):
            activations.append(backward_layer.activation)

        if len(activations) > 0:
            attrs.update(extract_activations(activations))

    else:
        attrs['direction'] = 'reverse' if op.go_backwards else 'forward'
        attrs['hidden_size'] = op.units

        if hasattr(op, 'activation'):
            attrs.update(extract_activations([op.activation]))

    return attrs


def build_output(scope, operator, container, output_names, bidirectional=False):
    """Builds the output stages for the RNN.
    """
    # TODO: Add bidirectional support
    rnn_y, rnn_h = output_names

    op = operator.raw_operator
    _, seq_length, input_size = extract_input_shape(op)
    hidden_size = op.units
    output_seq = op.return_sequences
    output_state = op.return_state

    output_name = operator.outputs[0].full_name
    tranposed_y = scope.get_unique_variable_name(operator.full_name + '_y_transposed')

    if output_seq:
        perm = [1, 0, 2] if container.target_opset <= 5 else [2, 0, 1, 3]
        apply_transpose(scope, rnn_y, tranposed_y, container, perm=perm)
        apply_reshape(scope, tranposed_y, output_name, container,
                      desired_shape=[-1, seq_length, hidden_size])
    else:
        # Here we ingore ONNX RNN's first output because it's useless.
        apply_transpose(scope, rnn_h, tranposed_y, container, perm=[1, 0, 2])
        apply_reshape(scope, tranposed_y, output_name, container, desired_shape=[-1, hidden_size])

    if output_state:
        apply_reshape(scope, rnn_h, operator.outputs[1].full_name, container, desired_shape=[-1, hidden_size])


def convert_keras_simple_rnn(scope, operator, container, bidirectional=False):
    op = operator.raw_operator

    _name = name_func(scope, operator)

    if bidirectional:
        output_seq = op.forward_layer.return_sequences
    else:
        output_seq = op.return_sequences

    # Inputs
    rnn_x = _name('X')
    tensor_w, tensor_r, tensor_b = build_parameters(scope, operator, container, bidirectional)
    sequence_lengths = build_sequence_lengths(scope, operator, container)
    initial_h = build_initial_states(scope, operator, container, bidirectional)

    input_names = [
        rnn_x,
        tensor_w,
        tensor_r,
        tensor_b,
        sequence_lengths,
        initial_h,
    ]

    # Attributes
    attrs = build_attributes(scope, operator, container, bidirectional)

    # Outputs
    output_names = [_name('Y'), _name('Y_h')]

    # Transpose input values
    input_name = operator.inputs[0].full_name
    apply_transpose(scope, input_name, rnn_x, container, perm=[1, 0, 2])

    oopb = OnnxOperatorBuilder(container, scope)
    oopb.apply_op_with_output('apply_rnn',
                              input_names,
                              output_names,
                              name=op.name,
                              output_seq=output_seq,
                              **attrs)

    build_output(scope, operator, container, output_names, bidirectional)
