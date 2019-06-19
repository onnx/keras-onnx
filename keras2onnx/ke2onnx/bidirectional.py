###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

import collections
import numbers
import numpy as np
from ..common import cvtfunc
from ..common.onnx_ops import apply_transpose, apply_split, apply_reshape, apply_identity, OnnxOperatorBuilder
from ..proto import onnx_proto, keras
from .common import extract_recurrent_activation
LSTM = keras.layers.LSTM


def _calculate_keras_bidirectional_output_shapes(operator):
    op = operator.raw_operator
    if isinstance(op.output_shape[0], collections.Iterable):
        operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else 'None'
                                              for i in op.output_shape[0])
        if op.merge_mode is None:
            operator.outputs[1].type.shape = list(i if isinstance(i, numbers.Integral) else 'None'
                                                  for i in op.output_shape[1])
    else:
        operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else 'None' for i in op.output_shape)


@cvtfunc(shape_infer=_calculate_keras_bidirectional_output_shapes)
def convert_bidirectional(scope, operator, container):
    # Extract basic information and create aliases for some fields
    op = operator.raw_operator
    forward_layer = op.forward_layer
    backward_layer = op.backward_layer
    input_shape = op.get_input_shape_at(0)
    # TODO: Add a test case for list
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    input_size = input_shape[-1]
    seq_length = input_shape[-2]
    is_static_shape = seq_length is not None
    if not is_static_shape and container.target_opset < 9:
        raise ValueError('None seq_length is not supported in opset ' + str(container.target_opset))
    hidden_size = forward_layer.units
    output_seq = forward_layer.return_sequences
    output_state = forward_layer.return_state
    if output_state:
        raise ValueError('Keras Bidirectional cannot return hidden and cell states')
    if not isinstance(forward_layer, LSTM):
        raise TypeError('The bidirectional module only works with LSTM in Keras but we got %s' % type(forward_layer))

    # Extract the forward transformation matrix used to adjust input features
    W_x = np.empty(shape=(4, hidden_size, input_size))
    K_W_x = forward_layer.get_weights()[0].T  # This matrix is a concatenation of W[ifco] in Keras
    # Set up W_i
    W_x[0:] = K_W_x[0 * hidden_size:][:hidden_size]
    # Set up W_o
    W_x[1:] = K_W_x[3 * hidden_size:][:hidden_size]
    # Set up W_f
    W_x[2:] = K_W_x[1 * hidden_size:][:hidden_size]
    # Set up W_c
    W_x[3:] = K_W_x[2 * hidden_size:][:hidden_size]

    # Extract the forward transformation matrix used to adjust hidden state
    W_h = np.empty(shape=(4, hidden_size, hidden_size))
    K_W_h = forward_layer.get_weights()[1].T  # This matrix is a concatenation of R[ifco] in Keras
    # Set up W_i
    W_h[0:] = K_W_h[0 * hidden_size:][:hidden_size]
    # Set up W_o
    W_h[1:] = K_W_h[3 * hidden_size:][:hidden_size]
    # Set up W_f
    W_h[2:] = K_W_h[1 * hidden_size:][:hidden_size]
    # Set up W_c
    W_h[3:] = K_W_h[2 * hidden_size:][:hidden_size]

    # Bias vectors of forward layer
    b = None
    if forward_layer.use_bias:
        b = np.zeros(shape=(8, hidden_size))
        keras_b = forward_layer.get_weights()[2]  # This matrix is a concatenation of B[ifco] in Keras
        # Set up B_i
        b[0] = keras_b[0 * hidden_size:][:hidden_size]
        # Set up B_o
        b[1] = keras_b[3 * hidden_size:][:hidden_size]
        # Set up B_f
        b[2] = keras_b[1 * hidden_size:][:hidden_size]
        # Set up B_c
        b[3] = keras_b[2 * hidden_size:][:hidden_size]

    # Extract the backward transformation matrix used to adjust input features. Note that the weight format for the
    # backward layer is identical to that of the forward layer.
    W_x_back = np.empty(shape=(4, hidden_size, input_size))
    keras_W_x = backward_layer.get_weights()[0].T
    W_x_back[0:] = keras_W_x[0 * hidden_size:][:hidden_size]
    W_x_back[1:] = keras_W_x[3 * hidden_size:][:hidden_size]
    W_x_back[2:] = keras_W_x[1 * hidden_size:][:hidden_size]
    W_x_back[3:] = keras_W_x[2 * hidden_size:][:hidden_size]

    # Extract the backward transformation matrix used to adjust hidden state
    W_h_back = np.empty(shape=(4, hidden_size, hidden_size))
    keras_W_h = backward_layer.get_weights()[1].T
    W_h_back[0:] = keras_W_h[0 * hidden_size:][:hidden_size]
    W_h_back[1:] = keras_W_h[3 * hidden_size:][:hidden_size]
    W_h_back[2:] = keras_W_h[1 * hidden_size:][:hidden_size]
    W_h_back[3:] = keras_W_h[2 * hidden_size:][:hidden_size]

    # Bias vectors of backward layer
    b_back = None
    if backward_layer.use_bias:
        b_back = np.zeros(shape=(8, hidden_size))
        keras_b = backward_layer.get_weights()[2]
        b_back[0] = keras_b[0 * hidden_size:][:hidden_size]
        b_back[1] = keras_b[3 * hidden_size:][:hidden_size]
        b_back[2] = keras_b[1 * hidden_size:][:hidden_size]
        b_back[3] = keras_b[2 * hidden_size:][:hidden_size]

    if (b is None and b_back is not None) or (b is not None and b_back is None):
        raise ValueError('Bidirectional bias must be enabled (or disabled) for both forward and backward layers.')

    # Declare ONNX LSTM (bidirectional is naturally supported)
    lstm__type = 'LSTM'
    lstm_input_names = []
    lstm_output_names = []
    lstm_attrs = {'name': operator.full_name}

    # Reshape Keras input format into ONNX input format
    lstm_x_name = scope.get_unique_variable_name(operator.full_name + '_X')
    apply_transpose(scope, operator.inputs[0].full_name, lstm_x_name, container, perm=[1, 0, 2])
    lstm_input_names.append(lstm_x_name)

    # Allocate input transformation matrix in ONNX and add its name into LSTM input list
    tensor_w_name = scope.get_unique_variable_name(operator.full_name + '_W')
    container.add_initializer(tensor_w_name, onnx_proto.TensorProto.FLOAT,
                              [2, 4 * hidden_size, input_size], np.concatenate([W_x, W_x_back]).flatten())
    lstm_input_names.append(tensor_w_name)

    # Allocate hidden transformation matrix in ONNX and add its name into LSTM input list
    tensor_r_name = scope.get_unique_variable_name(operator.full_name + '_R')
    container.add_initializer(tensor_r_name, onnx_proto.TensorProto.FLOAT,
                              [2, 4 * hidden_size, hidden_size], np.concatenate([W_h, W_h_back]).flatten())
    lstm_input_names.append(tensor_r_name)

    # Add bias vectors at different places in the original LSTM if needed
    if b is not None:
        tensor_b_name = scope.get_unique_variable_name(operator.full_name + '_B')
        container.add_initializer(tensor_b_name, onnx_proto.TensorProto.FLOAT, [2, 8 * hidden_size],
                                  np.concatenate([b, b_back]).flatten())
        lstm_input_names.append(tensor_b_name)
    else:
        lstm_input_names.append('')  # the name of a non-existing optional variable is an empty string

    oopb = OnnxOperatorBuilder(container, scope)
    input_shape_tensor = oopb.add_node('Shape',
                                       [operator.input_full_names[0]],
                                        operator.inputs[0].full_name + '_input_shape_tensor')

    if container.target_opset >= 10:
        batch_indices_tensor = oopb.add_node('Slice',
                                             [input_shape_tensor,
                                              ('_start', oopb.int64, np.array([0], dtype='int64')),
                                              ('_end', oopb.int64, np.array([1], dtype='int64')),
                                              ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                              ],
                                             operator.inputs[0].full_name + '_batch_indices_tensor')

        seq_len_tensor = oopb.add_node('Slice',
                                      [input_shape_tensor,
                                       ('_start', oopb.int64, np.array([1], dtype='int64')),
                                       ('_end', oopb.int64, np.array([2], dtype='int64')),
                                       ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                       ],
                                      operator.inputs[0].full_name + '_seq_len_tensor')
    else:
        attrs = {'starts': [0], 'ends': [1], 'axes': [0]}
        batch_indices_tensor = oopb.add_node('Slice',
                                             [input_shape_tensor],
                                             operator.inputs[0].full_name + '_batch_indices_tensor', **attrs)

        attrs = {'starts': [1], 'ends': [2], 'axes': [0]}
        seq_len_tensor = oopb.add_node('Slice',
                                       [input_shape_tensor],
                                       operator.inputs[0].full_name + '_seq_len_tensor', **attrs)

    # sequence_lens, this input is not used when converting Keras Bidirectional.
    lstm_input_names.append('')

    if is_static_shape:
        # need the zero initializer to correct some engine shape inference bug.
        state_shape = (2, 1, hidden_size)
        initial_h_name = scope.get_unique_variable_name(operator.full_name + '_initial_h')
        container.add_initializer(initial_h_name, onnx_proto.TensorProto.FLOAT, state_shape,
                                  np.zeros(shape=state_shape).flatten())
        lstm_input_names.append(initial_h_name)
        initial_c_name = scope.get_unique_variable_name(operator.full_name + '_initial_c')
        container.add_initializer(initial_c_name, onnx_proto.TensorProto.FLOAT, state_shape,
                                  np.zeros(shape=state_shape).flatten())
        lstm_input_names.append(initial_c_name)
    else:
        attrs = {'axis': 0}
        batch_size_tensor = oopb.add_node('Concat',
                                      [('_a', oopb.int64, np.array([2], dtype='int64')),
                                       batch_indices_tensor,
                                       ('_b', oopb.int64, np.array([hidden_size], dtype='int64'))
                                       ],
                                      operator.inputs[0].full_name + '_state_shape_tensor', **attrs)

        state_constant_shape_h = oopb.add_node('ConstantOfShape',
                                               [batch_size_tensor],
                                               operator.inputs[0].full_name + '_state_shape_constant_h')
        state_constant_shape_c = oopb.add_node('ConstantOfShape',
                                               [batch_size_tensor],
                                               operator.inputs[0].full_name + '_state_shape_constant_c')
        lstm_input_names.append(state_constant_shape_h)
        lstm_input_names.append(state_constant_shape_c)

    # P (optional) : No peep hole in keras.
    lstm_input_names.append('')

    activation_types = []
    alphas = []
    betas = []
    extracted_activations = [
        extract_recurrent_activation(forward_layer.recurrent_activation),
        extract_recurrent_activation(forward_layer.activation),
        extract_recurrent_activation(forward_layer.activation),
        extract_recurrent_activation(backward_layer.recurrent_activation),
        extract_recurrent_activation(backward_layer.activation),
        extract_recurrent_activation(backward_layer.activation)]

    for (activation_type, alpha, beta) in extracted_activations:
        activation_types.append(activation_type.encode('utf-8'))
        if alpha is not None:
            alphas.append(alpha)
        if beta is not None:
            betas.append(beta)

    lstm_attrs['activations'] = activation_types
    if alphas:
        lstm_attrs['activation_alpha'] = alphas
    if betas:
        lstm_attrs['activation_beta'] = betas

    lstm_attrs['direction'] = 'bidirectional'
    lstm_attrs['hidden_size'] = hidden_size
    # Set up version-dependent attributes
    if container.target_opset <= 5:
        lstm_attrs['output_sequence'] = 1 if output_seq else 0
        op_version = 1
    else:
        op_version = 7

    if hasattr(op, 'merge_mode'):
        if op.merge_mode not in ['concat', None]:
            raise ValueError('Only support Bidirectional with merge_mode=\'concat\' but got %s' % op.merge_mode)
        merge_concat = False if op.merge_mode is None else True
    else:
        merge_concat = False

    # Create variable names to store ONNX LSTM outputs. Those outputs need to be adjusted to meet the original Keras
    # LSTM behavior.
    lstm_y_name = scope.get_unique_variable_name(operator.full_name + '_Y')
    lstm_h_name = scope.get_unique_variable_name(operator.full_name + '_Y_h')
    lstm_c_name = scope.get_unique_variable_name(operator.full_name + '_Y_c')
    lstm_output_names.append(lstm_y_name)
    lstm_output_names.append(lstm_h_name)
    lstm_output_names.append(lstm_c_name)

    # Create the major node, ONNX LSTM
    container.add_node('LSTM', lstm_input_names, lstm_output_names, op_version=op_version, **lstm_attrs)

    if output_seq:
        # The output shape of runtime is 3-D while ONNX says 4-D, so we do a Reshape to fix it.
        if is_static_shape:
            lstm_y_name_fixed = scope.get_unique_variable_name(operator.full_name + '_Y_fixed')
            apply_reshape(scope, lstm_y_name, lstm_y_name_fixed, container, desired_shape=[seq_length, 2, -1, hidden_size])
        else:
            attrs = {'axis': 0}
            shape_tensor = oopb.add_node('Concat',
                                         [seq_len_tensor,
                                          ('_a', oopb.int64, np.array([2], dtype='int64')),
                                          ('_b', oopb.int64, np.array([-1], dtype='int64')),
                                          ('_c', oopb.int64, np.array([hidden_size], dtype='int64'))
                                          ],
                                         operator.inputs[0].full_name + '_output_seq_shape', **attrs)
            lstm_y_name_fixed = oopb.add_node('Reshape',
                                             [lstm_y_name,
                                              shape_tensor
                                              ],
                                             operator.inputs[0].full_name + '_output_seq_shape_1')

        attrs = {'axis': 0}
        shape_tensor_2 = oopb.add_node('Concat',
                                     [('_a', oopb.int64, np.array([-1], dtype='int64')),
                                       seq_len_tensor,
                                      ('_b', oopb.int64, np.array([2 * hidden_size], dtype='int64'))
                                      ],
                                     operator.inputs[0].full_name + '_output_seq_shape_2', **attrs)
        if merge_concat:
            # In this case, only one Keras output with shape (N, T, 2 * C') should be produced

            # Transpose ONNX LSTM Y with shape (T, D, N, C') into (T, N, D, C')
            transposed_y_name = scope.get_unique_variable_name(operator.full_name + '_Y_transposed')
            apply_transpose(scope, lstm_y_name_fixed, transposed_y_name, container, perm=[0, 2, 1, 3])

            # Change shape (T, N, D, C') to (N, T, D * C') to meet Keras spec
            if is_static_shape:
                apply_reshape(scope, transposed_y_name, operator.outputs[0].full_name, container,
                              desired_shape=[-1, seq_length, 2 * hidden_size])
            else:
                shape_tensor_output = oopb.add_node('Reshape',
                                                  [transposed_y_name,
                                                   shape_tensor_2
                                                   ],
                                                  operator.inputs[0].full_name + '_output_merge_concat')
                apply_identity(scope, shape_tensor_output, operator.outputs[0].full_name, container)
        else:
            # If merge_mode=None, two tensors should be generated. The first/second tensor is the output of
            # forward/backward pass.

            # Transpose ONNX LSTM Y with shape (T, D, N, C') into (T, N, D, C')
            transposed_y_name = scope.get_unique_variable_name(operator.full_name + '_Y_transposed')
            apply_transpose(scope, lstm_y_name_fixed, transposed_y_name, container, perm=[0, 2, 1, 3])

            # Split the transposed Y with shape (T, N, D, C') into (T, N, 1, C') and (T, N, 1, C')
            forward_y_name = scope.get_unique_variable_name(operator.full_name + '_Y_forward')
            backward_y_name = scope.get_unique_variable_name(operator.full_name + '_Y_backward')
            apply_split(scope, transposed_y_name, [forward_y_name, backward_y_name], container, axis=2)

            # Change (T, N, 1, C') into (T, N, C') to meet Keras spec
            forward_y_name_1 = scope.get_unique_variable_name(operator.full_name + '_Y_forward_1')
            backward_y_name_1 = scope.get_unique_variable_name(operator.full_name + '_Y_backward_1')
            container.add_node('Squeeze', forward_y_name, forward_y_name_1,
                               name=scope.get_unique_variable_name('Squeeze'), axes=[2])
            container.add_node('Squeeze', backward_y_name, backward_y_name_1,
                               name=scope.get_unique_variable_name('Squeeze'), axes=[2])

            if is_static_shape:
                apply_reshape(scope, forward_y_name_1, operator.outputs[0].full_name, container,
                              desired_shape=[-1, seq_length, hidden_size])
                apply_reshape(scope, backward_y_name_1, operator.outputs[1].full_name, container,
                              desired_shape=[-1, seq_length, hidden_size])
            else:
                shape_tensor_3 = oopb.add_node('Concat',
                                               [('_a', oopb.int64, np.array([-1], dtype='int64')),
                                                seq_len_tensor,
                                                ('_b', oopb.int64, np.array([hidden_size], dtype='int64'))
                                                ],
                                               operator.inputs[0].full_name + '_output_seq_shape_3', **attrs)
                shape_tensor_output_0 = oopb.add_node('Reshape',
                                                  [forward_y_name_1,
                                                   shape_tensor_3
                                                   ],
                                                  operator.inputs[0].full_name + '_shape_tensor_output_0')
                shape_tensor_output_1 = oopb.add_node('Reshape',
                                                  [backward_y_name_1,
                                                   shape_tensor_3
                                                   ],
                                                  operator.inputs[0].full_name + '_shape_tensor_output_1')
                apply_identity(scope, shape_tensor_output_0, operator.outputs[0].full_name, container)
                apply_identity(scope, shape_tensor_output_1, operator.outputs[1].full_name, container)
    else:
        perm = [1, 0, 2]
        if merge_concat:
            # In this case, only one Keras output with shape (N, 2 * C') should be produced

            # Transpose ONNX LSTM Y_h with shape (D, N, C') into (N, D, C')
            transposed_h_name = scope.get_unique_variable_name(operator.full_name + '_Y_h_transposed')
            apply_transpose(scope, lstm_h_name, transposed_h_name, container, perm=perm)

            # Maintain backwards opset compatibility for 'Flatten'
            op_version = 1 if container.target_opset < 9 else 9

            # Flatten ONNX (N, D, C') into (N, D * C')
            container.add_node('Flatten', transposed_h_name, operator.outputs[0].full_name,
                               name=scope.get_unique_variable_name('Flatten'), axis=1, op_version=op_version)
        else:
            # If merge_mode=None, two tensors should be generated. The first/second tensor is the output of
            # forward/backward pass.

            # Transpose ONNX LSTM Y_h with shape (D, N, C') into (N, D, C')
            transposed_h_name = scope.get_unique_variable_name(operator.full_name + '_Y_h_transposed')
            apply_transpose(scope, lstm_h_name, transposed_h_name, container, perm=perm)

            # Split the transposed Y with shape (T, N, D, C') into (T, N, 1, C') and (T, N, 1, C')
            forward_y_name = scope.get_unique_variable_name(operator.full_name + '_Y_forward')
            backward_y_name = scope.get_unique_variable_name(operator.full_name + '_Y_backward')
            apply_split(scope, transposed_h_name, [forward_y_name, backward_y_name], container, axis=2)

            # Change (T, N, 1, C') into (T, N, C') to meet Keras spec
            container.add_node('Squeeze', forward_y_name, operator.outputs[0].full_name,
                               name=scope.get_unique_variable_name('Squeeze'), axes=[2])
            container.add_node('Squeeze', backward_y_name, operator.outputs[1].full_name,
                               name=scope.get_unique_variable_name('Squeeze'), axes=[2])
