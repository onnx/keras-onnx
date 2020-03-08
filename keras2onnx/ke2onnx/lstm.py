###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numbers
import numpy as np
from collections.abc import Iterable
from ..common import cvtfunc
from ..common.onnx_ops import apply_transpose, apply_reshape, apply_identity, OnnxOperatorBuilder
from ..proto import onnx_proto
from .common import extract_recurrent_activation
from . import simplernn



def convert_ifco_to_iofc(tensor_ifco):
    """Returns a tensor in input (i), output (o), forget (f), cell (c) ordering. The
    Keras ordering is ifco, while the ONNX ordering is iofc.
    """
    splits = np.split(tensor_ifco, 4)
    return np.concatenate((splits[0], splits[3], splits[1], splits[2]))


def extract_params(op, hidden_size, input_size):
    """Returns a tuple of the LSTM parameters, and converts them into the format for ONNX.
    """
    params = op.get_weights()

    # Keras: [W_x, W_h, b] each in I F C O
    # ONNX: W[iofc] I O F C
    W_x = convert_ifco_to_iofc(params[0].T).reshape(4, hidden_size, input_size)
    W_h = convert_ifco_to_iofc(params[1].T).reshape(4, hidden_size, hidden_size)

    b = None
    if op.use_bias:
        b = np.zeros((8, hidden_size), dtype=np.float32)
        b[:4] = convert_ifco_to_iofc(params[2]).reshape(4, hidden_size)

    return W_x, W_h, b


def _calculate_keras_lstm_output_shapes(operator):
    op = operator.raw_operator
    if isinstance(op.output_shape[0], Iterable):
        operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else None
                                              for i in op.output_shape[0])
    else:
        operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else None for i in op.output_shape)


@cvtfunc(shape_infer=_calculate_keras_lstm_output_shapes)
def convert_keras_lstm(scope, operator, container):
    op = operator.raw_operator
    hidden_size = op.units
    _, seq_length, input_size = simplernn.extract_input_shape(op)

    W_x, W_h, b = extract_params(op, hidden_size, input_size)

    is_static_shape = seq_length is not None
    if not is_static_shape and container.target_opset < 9:
        raise ValueError('None seq_length is not supported in opset ' + str(container.target_opset))
    output_seq = op.return_sequences
    output_state = op.return_state
    reverse_input = op.go_backwards

    # Declare essential attributes of ONNX LSTM
    lstm_input_names = []
    lstm_output_names = []
    lstm_attrs = {}

    # Because of the format difference between Keras and ONNX LSTM's, we set up a preprocessing node to match them.
    lstm_x_name = scope.get_unique_variable_name('lstm_x')
    lstm_input_names.append(lstm_x_name)
    apply_transpose(scope, operator.inputs[0].full_name, lstm_x_name, container, perm=[1, 0, 2])

    # Add the weights to the final model's initializer list so that our LSTM operator can use it
    tensor_w_name = scope.get_unique_variable_name('W')
    container.add_initializer(tensor_w_name, onnx_proto.TensorProto.FLOAT,
                              [1, 4 * hidden_size, input_size], W_x.flatten())
    lstm_input_names.append(tensor_w_name)

    # Add the recursion weights to the final model's initializer list so that our LSTM operator can use it
    tensor_r_name = scope.get_unique_variable_name('R')
    container.add_initializer(tensor_r_name, onnx_proto.TensorProto.FLOAT,
                              [1, 4 * hidden_size, hidden_size], W_h.flatten())
    lstm_input_names.append(tensor_r_name)

    if b is not None and len(b) > 0:
        tensor_b_name = scope.get_unique_variable_name('B')
        container.add_initializer(tensor_b_name, onnx_proto.TensorProto.FLOAT, [1, 8 * hidden_size], b.flatten())
        lstm_input_names.append(tensor_b_name)
    else:
        lstm_input_names.append('')

    # sequence_lens
    uses_masking_layer = len(operator.input_masks) == 1
    if uses_masking_layer:
        # Mask using sequence_lens input
        sequence_lengths = scope.get_unique_variable_name(operator.full_name + '_seq_lens')
        lstm_input_names.append(sequence_lengths)
    else:
        lstm_input_names.append('')
    # inital_h
    if len(operator.inputs) <= 1:
        lstm_input_names.append('')
    else:
        # Add a reshape after initial_h, 2d -> 3d
        inital_h_reshape = scope.get_unique_variable_name('inital_h_reshape')
        apply_reshape(scope, operator.inputs[1].full_name, inital_h_reshape, container,
                      desired_shape=[1, -1, hidden_size])
        lstm_input_names.append(inital_h_reshape)
    # initial_c
    if len(operator.inputs) <= 2:
        lstm_input_names.append('')
    else:
        # Add a reshape after initial_h, 2d -> 3d
        inital_c_reshape = scope.get_unique_variable_name('inital_c_reshape')
        apply_reshape(scope, operator.inputs[2].full_name, inital_c_reshape, container,
                      desired_shape=[1, -1, hidden_size])
        lstm_input_names.append(inital_c_reshape)
    # P (optional) : No peep hole in keras.
    lstm_input_names.append('')

    activation_types = []
    alphas = []
    betas = []
    extracted_activations = [
        extract_recurrent_activation(op.recurrent_activation),
        extract_recurrent_activation(op.activation),
        extract_recurrent_activation(op.activation)]

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

    # Set up other attributes
    lstm_attrs['direction'] = 'reverse' if reverse_input else 'forward'
    lstm_attrs['hidden_size'] = hidden_size

    # We declare some names to store the outputs produced by ONNX LSTM. Then, create ONNX LSTM. Subsequently, its
    # outputs may be adjusted to match Keras format.
    lstm_y_name = scope.get_unique_variable_name('lstm_y')
    lstm_output_names.append(lstm_y_name)
    lstm_h_name = scope.get_unique_variable_name('lstm_h')
    lstm_output_names.append(lstm_h_name)
    lstm_c_name = scope.get_unique_variable_name('lstm_c')
    lstm_output_names.append(lstm_c_name)

    oopb = OnnxOperatorBuilder(container, scope)

    if uses_masking_layer:
        mask_cast = oopb.apply_cast(operator.input_masks[0].full_name, to=oopb.int32, name=operator.full_name + '_mask_cast')
        oopb.add_node_with_output('ReduceSum', mask_cast, sequence_lengths, keepdims=False, axes=[-1], name=operator.full_name + '_mask_sum')

    oopb.apply_op_with_output('apply_lstm',
                              lstm_input_names,
                              lstm_output_names,
                              name=operator.raw_operator.name,
                              output_seq=output_seq,
                              **lstm_attrs)

    # Create output-adjusting operators
    if output_seq:
        lstm_y_name_transposed = scope.get_unique_variable_name('lstm_y_transposed')
        perm = [1, 0, 2] if container.target_opset <= 5 else [2, 0, 1, 3]
        apply_transpose(scope, lstm_y_name, lstm_y_name_transposed, container, perm=perm)
        if is_static_shape:
            apply_reshape(scope, lstm_y_name_transposed, operator.outputs[0].full_name, container,
                          desired_shape=[-1, seq_length, hidden_size])
        else:
            input_shape_tensor = oopb.add_node('Shape',
                                               [operator.input_full_names[0]],
                                               operator.inputs[0].full_name + '_input_shape_tensor')

            if container.target_opset >= 10:
                seq_len_tensor = oopb.add_node('Slice',
                                               [input_shape_tensor,
                                                ('_start', oopb.int64, np.array([1], dtype='int64')),
                                                ('_end', oopb.int64, np.array([2], dtype='int64')),
                                                ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                                ],
                                               operator.inputs[0].full_name + '_seq_len_tensor')
            else:
                seq_len_tensor = oopb.add_node('Slice',
                                               [input_shape_tensor],
                                               operator.inputs[0].full_name + '_seq_len_tensor', starts=[1], ends=[2],
                                               axes=[0])

            shape_tensor = oopb.add_node('Concat',
                                         [('_a', oopb.int64, np.array([-1], dtype='int64')),
                                          seq_len_tensor,
                                          ('_b', oopb.int64, np.array([hidden_size], dtype='int64'))
                                          ],
                                         operator.inputs[0].full_name + '_output_seq_shape', axis=0)
            shape_tensor_output = oopb.add_node('Reshape',
                                                [lstm_y_name_transposed,
                                                 shape_tensor
                                                ],
                                                operator.inputs[0].full_name + '_output_seq_shape_1')
            apply_identity(scope, shape_tensor_output, operator.outputs[0].full_name, container)
    else:
        apply_reshape(scope, lstm_h_name, operator.outputs[0].full_name, container, desired_shape=[-1, hidden_size])

    if output_state:
        # state_h
        apply_reshape(scope, lstm_h_name, operator.outputs[1].full_name, container, desired_shape=[-1, hidden_size])
        # state_c
        apply_reshape(scope, lstm_c_name, operator.outputs[2].full_name, container, desired_shape=[-1, hidden_size])
