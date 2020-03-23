###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numpy as np
from ..proto import onnx_proto
from ..common import name_func
from ..common.onnx_ops import apply_transpose, OnnxOperatorBuilder
from . import simplernn

TensorProto = onnx_proto.TensorProto


def extract_params(op):
    """Returns a tuple of the GRU paramters, and converts them into the format for ONNX.
    """
    params = op.get_weights()
    W = params[0].T
    R = params[1].T

    B = None
    if op.use_bias:
        B = params[2]

    return W, R, B

def build_parameters(scope, operator, container, bidirectional=False):
    """Returns the parameter initialization values after extracting them from the GRU layer.
    """
    op = operator.raw_operator
    _, seq_length, input_size = simplernn.extract_input_shape(op)

    _name = name_func(scope, operator)

    tensor_w = _name('W')
    tensor_r = _name('R')
    tensor_b = ''

    if bidirectional:
        forward_layer = op.forward_layer
        backward_layer = op.backward_layer
        hidden_size = forward_layer.units

        W, R, B = extract_params(forward_layer)
        W_back, R_back, B_back = extract_params(backward_layer)

        W = np.concatenate([W, W_back])
        W_shape = [2, 3 * hidden_size, input_size]

        R = np.concatenate([R, R_back])
        R_shape = [2, 3 * hidden_size, hidden_size]

        if B is not None:
            if B.size == 3 * hidden_size:
                B = np.concatenate([B, np.zeros(3 * hidden_size)])
            if B_back.size == 3 * hidden_size:
                B_back = np.concatenate([B_back, np.zeros(3 * hidden_size)])
            B = np.concatenate([B, B_back])
            B_shape = [2, 6 * hidden_size]

    else:
        hidden_size = op.units

        W, R, B = extract_params(op)
        W_shape = [1, 3 * hidden_size, input_size]
        R_shape = [1, 3 * hidden_size, hidden_size]

        if B is not None:
            if B.size == 3 * hidden_size:
                B = np.concatenate([B, np.zeros(3 * hidden_size)])
            B_shape = [1, 6 * hidden_size]

    # Create initializers
    container.add_initializer(tensor_w, TensorProto.FLOAT, W_shape, W.flatten())
    container.add_initializer(tensor_r, TensorProto.FLOAT, R_shape, R.flatten())

    if B is not None:
        tensor_b = _name('B')
        container.add_initializer(tensor_b, TensorProto.FLOAT, B_shape, B.flatten())

    return tensor_w, tensor_r, tensor_b


def build_attributes(scope, operator, container, bidirectional=False):
    """Returns a dictionary of attributes for the GRU layer.
    """
    op = operator.raw_operator

    attrs = {}

    if bidirectional:
        forward_layer = op.forward_layer
        backward_layer = op.backward_layer

        attrs['direction'] = 'bidirectional'
        attrs['hidden_size'] = forward_layer.units
        attrs.update(simplernn.extract_activations([
            forward_layer.recurrent_activation,
            forward_layer.activation,
            backward_layer.recurrent_activation,
            backward_layer.activation,

        ]))

    else:
        attrs['direction'] = 'reverse' if op.go_backwards else 'forward'
        attrs['hidden_size'] = op.units
        attrs.update(simplernn.extract_activations([
            op.recurrent_activation,
            op.activation
        ]))

    return attrs


def convert_keras_gru(scope, operator, container, bidirectional=False):
    op = operator.raw_operator

    _name = name_func(scope, operator)

    if bidirectional:
        output_seq = op.forward_layer.return_sequences
    else:
        output_seq = op.return_sequences

    # Inputs
    gru_x = _name('X')
    tensor_w, tensor_r, tensor_b = build_parameters(scope, operator, container, bidirectional)
    sequence_lengths = simplernn.build_sequence_lengths(scope, operator, container)
    initial_h = simplernn.build_initial_states(scope, operator, container, bidirectional)

    input_names = [
        gru_x,
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
    apply_transpose(scope, input_name, gru_x, container, perm=[1, 0, 2])

    oopb = OnnxOperatorBuilder(container, scope)
    oopb.apply_op_with_output('apply_gru',
                              input_names,
                              output_names,
                              name=op.name,
                              output_seq=output_seq,
                              reset_after=op.reset_after,
                              **attrs)

    simplernn.build_output(scope, operator, container, output_names, bidirectional)
