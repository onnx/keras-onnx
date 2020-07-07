###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numpy as np
from ..proto import onnx_proto, keras
from ..common import name_func
from ..common.onnx_ops import (
    apply_cast,
    apply_concat,
    apply_reshape,
    apply_slice,
    apply_split,
    apply_squeeze,
    apply_transpose,
    apply_unsqueeze,
    OnnxOperatorBuilder,
)

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
    elif activation == activations.softsign:
        onnx_op_type = 'Softsign'
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

    op = operator.raw_operator
    _name = name_func(scope, operator)

    initial_h = _name('initial_h')

    if bidirectional:
        forward_layer = op.forward_layer
        hidden_size = forward_layer.units
        desired_shape = [1, -1, hidden_size]

        # Combine the forward and backward layers
        forward_h = _name('initial_h_forward')
        backward_h = _name('initial_h_backward')

        # Handle LSTM initial hidden case to enable code reuse
        if len(operator.inputs) > 4:
            f, b = 1, 3
        else:
            f, b = 1, 2

        apply_reshape(scope, operator.inputs[f].full_name, forward_h, container, desired_shape=desired_shape)
        apply_reshape(scope, operator.inputs[b].full_name, backward_h, container, desired_shape=desired_shape)

        apply_concat(scope, [forward_h, backward_h], initial_h, container)

    else:
        # Unsqueeze dim 0 to represent num_directions
        input_h = operator.inputs[1].full_name
        apply_unsqueeze(scope, input_h, initial_h, container, axes=[0])
    return initial_h


def build_attributes(scope, operator, container, bidirectional=False):
    """Returns a dictionary of attributes for the RNN layer.
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
    rnn_y, rnn_h = output_names

    op = operator.raw_operator
    _, seq_length, input_size = extract_input_shape(op)
    is_static_shape = seq_length is not None

    _name = name_func(scope, operator)

    oopb = OnnxOperatorBuilder(container, scope)

    # Define seq_dim
    if not is_static_shape:
        input_name = operator.inputs[0].full_name
        input_shape_tensor = oopb.add_node('Shape', [input_name], input_name + '_input_shape_tensor')

        seq_dim = input_name + '_seq_dim'
        apply_slice(scope, input_shape_tensor, seq_dim, container, [1], [2], axes=[0])

    if bidirectional:
        time_major = is_time_major(op, bidirectional)
        forward_layer = op.forward_layer

        hidden_size = forward_layer.units
        output_seq = forward_layer.return_sequences

        merge_concat = False
        if hasattr(op, 'merge_mode'):
            if op.merge_mode not in ['concat', None]:
                raise ValueError('Bidirectional only supports merge_mode=\'concat\' '
                                 'but got %s' % op.merge_mode)
            if op.merge_mode is not None:
                merge_concat = True

        if output_seq:
            lstm_out = _name('y_transposed')
            if not time_major:
                # Transpose ONNX RNN Y with shape (T, D, N, C') into (N, T, D, C')
                apply_transpose(scope, rnn_y, lstm_out, container, perm=[2, 0, 1, 3])
            else:
                # Transpose RNN Y with shape (T, D, N, C) into (T, N, D, C)
                apply_transpose(scope, rnn_y, lstm_out, container, perm=[0, 2, 1, 3])
            if merge_concat:
                # In this case, only one Keras output with shape (N, T, 2 * C') should be produced.
                # ( T, N, 2*C ) if it was time major.
                apply_reshape(scope, lstm_out, operator.outputs[0].full_name, container,
                              desired_shape=[0, 0, 2 * hidden_size])
            else:
                # If merge_mode=None, two tensors should be generated. The first/second tensor is the output of
                # forward/backward pass.

                # Split the transposed Y with shape (T, N, D, C') into (T, N, 1, C') and (T, N, 1, C')
                forward_y = _name('Y_forward')
                backward_y = _name('Y_backward')
                axis_direction = 2
                apply_split(scope, lstm_out, [forward_y, backward_y], container, axis=axis_direction)

                # Change (T, N, 1, C') into (T, N, C') to meet Keras spec
                apply_squeeze(scope, forward_y, operator.outputs[0].full_name, container, axes=[axis_direction])
                apply_squeeze(scope, backward_y, operator.outputs[1].full_name, container, axes=[axis_direction])
        else:
            perm = [1, 0, 2]
            if merge_concat:
                # In this case, only one Keras output with shape (N, 2 * C') should be produced

                # Transpose ONNX RNN Y_h with shape (D, N, C') into (N, D, C')
                transposed_h = _name('Y_h_transposed')
                apply_transpose(scope, rnn_h, transposed_h, container, perm=perm)

                # Flatten ONNX (N, D, C') into (N, D * C')
                oopb.apply_op_with_output("apply_flatten",
                                          transposed_h,
                                          operator.outputs[0].full_name,
                                          name=operator.full_name + '_flatten',
                                          axis=1)
            else:
                # If merge_mode=None, two tensors should be generated. The first/second tensor is the output of
                # forward/backward pass.

                # Transpose ONNX RNN Y_h with shape (D, N, C') into (N, D, C')
                transposed_h = _name('Y_h_transposed')
                apply_transpose(scope, rnn_h, transposed_h, container, perm=perm)

                # Split the transposed Y with shape (T, N, D, C') into (T, N, 1, C') and (T, N, 1, C')
                forward_y = _name('Y_forward')
                backward_y = _name('Y_backward')
                axis_direction = 1
                apply_split(scope, transposed_h, [forward_y, backward_y], container, axis=axis_direction)

                # Change (T, N, 1, C') into (T, N, C') to meet Keras spec
                apply_squeeze(scope, forward_y, operator.outputs[0].full_name, container, axes=[axis_direction])
                apply_squeeze(scope, backward_y, operator.outputs[1].full_name, container, axes=[axis_direction])
    else:
        hidden_size = op.units
        output_seq = op.return_sequences

        output_name = operator.outputs[0].full_name
        transposed_y = scope.get_unique_variable_name(operator.full_name + '_y_transposed')

        # Determine the source, transpose permutation, and output shape
        if output_seq:
            source = rnn_y
            perm = [2, 0, 1, 3]
            if is_static_shape:
                desired_shape = [-1, seq_length, hidden_size]
            elif container.target_opset < 5:
                # Before Reshape-5 you can not take the sequence dimension in as an input
                raise ValueError('At least opset 5 is required for output sequences')
            else:
                # Dynamically determine the output shape based on the sequence dimension
                shape_values = [
                    ('_a', oopb.int64, np.array([-1], dtype='int64')),
                    seq_dim,
                    ('_b', oopb.int64, np.array([hidden_size], dtype='int64')),
                ]
                shape_name = _name('_output_seq_shape')
                desired_shape = oopb.add_node('Concat', shape_values, shape_name, axis=0)
        else:
            # Use the last hidden states directly
            source = rnn_h
            perm = [1, 0, 2]
            desired_shape = [-1, hidden_size]

        apply_transpose(scope, source, transposed_y, container, perm=perm)
        apply_reshape(scope, transposed_y, output_name, container, desired_shape=desired_shape)


def build_output_states(scope, operator, container, output_names, bidirectional=False):
    """Builds the output hidden states for the RNN layer.
    """
    _, rnn_h = output_names
    op = operator.raw_operator

    if bidirectional:
        forward_layer = op.forward_layer
        output_state = forward_layer.return_state

        if output_state:
            # Split rnn_h into forward and backward directions
            output_names = [o.full_name for o in operator.outputs[1:]]
            split_names = ['{}_{}'.format(rnn_h, d) for d in ('forward', 'backward')]

            apply_split(scope, rnn_h, split_names, container)

            for split_name, output_name in zip(split_names, output_names):
                apply_squeeze(scope, split_name, output_name, container)

    else:
        output_state = op.return_state

        if output_state:
            output_h = operator.outputs[1].full_name
            apply_squeeze(scope, rnn_h, output_h, container)


def is_time_major(op, bidirectional):
    if bidirectional:
        time_major = op.forward_layer.time_major if hasattr(op.forward_layer, "time_major") else False
    else:
        time_major = op.time_major if hasattr(op, "time_major") else False
    return time_major


def convert_keras_simple_rnn(scope, operator, container, bidirectional=False):
    op = operator.raw_operator

    _name = name_func(scope, operator)

    if bidirectional:
        output_seq = op.forward_layer.return_sequences
    else:
        output_seq = op.return_sequences
    time_major = is_time_major(op, bidirectional)

    # Inputs
    rnn_x = operator.inputs[0].full_name
    if not time_major:
        rnn_x = _name('X')
        apply_transpose(scope, operator.inputs[0].full_name, rnn_x, container, perm=[1, 0, 2])
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

    oopb = OnnxOperatorBuilder(container, scope)
    oopb.apply_op_with_output('apply_rnn',
                              input_names,
                              output_names,
                              name=op.name,
                              output_seq=output_seq,
                              **attrs)

    build_output(scope, operator, container, output_names, bidirectional)
    build_output_states(scope, operator, container, output_names, bidirectional)
