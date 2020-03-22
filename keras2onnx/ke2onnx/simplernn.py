###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numpy as np
from ..proto import onnx_proto, keras
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


def build_initial_states(scope, operator, container):
    """Reshapes the initial input states. If there are no states present as inputs, then
    it returns an empty input for the initial hidden states.
    """
    # Initial hidden states
    if len(operator.inputs) == 1:
        return ''

    # Add a reshape after initial_h, 2d -> 3d
    input_h_name = operator.inputs[1].full_name
    initial_h_name = get_name('_initial_h')
    apply_reshape(scope, input_h_name, initial_h_name, container, desired_shape=[1, -1, hidden_size])
    return initial_h_name


def convert_keras_simple_rnn(scope, operator, container):
    op = operator.raw_operator
    hidden_size = op.units
    _, seq_length, input_size = extract_input_shape(op)
    output_seq = op.return_sequences
    output_state = op.return_state
    reverse_input = op.go_backwards

    input_name = operator.inputs[0].full_name
    get_name = lambda x: scope.get_unique_variable_name(operator.full_name + x)

    # Inputs
    rnn_x_name = get_name('_X')
    tensor_w_name = get_name('_W')
    tensor_r_name = get_name('_R')
    tensor_b_name = ''
    sequence_lengths = build_sequence_lengths(scope, operator, container)
    initial_h_name = build_initial_states(scope, operator, container)

    W, R, B = extract_params(op, hidden_size)
    W_shape = [1, hidden_size, input_size]
    R_shape = [1, hidden_size, hidden_size]

    container.add_initializer(tensor_w_name, TensorProto.FLOAT, W_shape, W.flatten())
    container.add_initializer(tensor_r_name, TensorProto.FLOAT, R_shape, R.flatten())

    if op.use_bias:
        tensor_b_name = get_name('_B')
        B_shape = [1, 2 * hidden_size]
        container.add_initializer(tensor_b_name, TensorProto.FLOAT, B_shape, B.flatten())

    input_names = [
        rnn_x_name,
        tensor_w_name,
        tensor_r_name,
        tensor_b_name,
        sequence_lengths,
        initial_h_name,
    ]

    attrs = {}
    if hasattr(op, 'activation'):
        attrs.update(extract_activations([op.activation]))

    attrs['direction'] = 'reverse' if reverse_input else 'forward'
    attrs['hidden_size'] = hidden_size

    # We use the collected information to build ONNX's RNN. ONNX RNN's outputs will be saved onto two intermediate
    # tensors and we will adjust them subsequently to mimic Keras output format.
    rnn_y_name = get_name('_y')
    rnn_h_name = get_name('_h')

    output_names = [
        rnn_y_name,
        rnn_h_name,
    ]


    apply_transpose(scope, input_name, rnn_x_name, container, perm=[1, 0, 2])

    oopb = OnnxOperatorBuilder(container, scope)
    oopb.apply_op_with_output('apply_rnn',
                              input_names,
                              output_names,
                              name=operator.raw_operator.name,
                              output_seq=output_seq,
                              **attrs)

    # Create operators to adjust ONNX output to meet Keras format
    if output_seq:
        permuted_rnn_y_name = get_name('_y_permuted')
        perm = [1, 0, 2] if container.target_opset <= 5 else [2, 0, 1, 3]
        apply_transpose(scope, rnn_y_name, permuted_rnn_y_name, container, perm=perm)
        apply_reshape(scope, permuted_rnn_y_name, operator.outputs[0].full_name, container,
                      desired_shape=[-1, seq_length, hidden_size])
    else:
        # Here we ingore ONNX RNN's first output because it's useless.
        apply_reshape(scope, rnn_h_name, operator.outputs[0].full_name, container, desired_shape=[-1, hidden_size])

    if output_state:
        apply_reshape(scope, rnn_h_name, operator.outputs[1].full_name, container, desired_shape=[-1, hidden_size])
