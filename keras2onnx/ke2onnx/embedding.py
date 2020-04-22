###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from ..common.onnx_ops import apply_reshape, apply_cast, OnnxOperatorBuilder
from ..proto import onnx_proto

import numpy as np


def convert_keras_embed(scope, operator, container):
    op = operator.raw_operator  # Keras Embedding layer object
    #  if mask_zero specified, the output_mask tensor needed by calculated
    if hasattr(op, 'mask_zero') and op.mask_zero is True:
        oopb = OnnxOperatorBuilder(container, scope)
        # Keras embed layer compute mask
        # output_mask = K.not_equal(inputs, 0)
        if container.target_opset >= 11:
            equal_out = oopb.add_node('Equal', [operator.inputs[0].full_name, np.array([0], dtype='float32')],
                                      operator.full_name + 'mask')
            container.add_node('Not', equal_out, operator.output_masks[0].full_name,
                               name=operator.full_name + 'mask_not')
        else:
            equal_input_0 = oopb.add_node('Cast', [operator.inputs[0].full_name],
                                          operator.full_name + '_input_cast', to=6)
            equal_out = oopb.add_node('Equal', [equal_input_0, np.array([0], dtype='int32')],
                                      operator.full_name + 'mask')
            container.add_node('Not', equal_out, operator.output_masks[0].full_name,
                               name=operator.full_name + 'mask_not')

    cast_name = scope.get_unique_variable_name('casted')
    apply_cast(scope, operator.inputs[0].full_name, cast_name, container, to=onnx_proto.TensorProto.INT32)

    # Prepare the weight matrix (i.e., the vectors of all input indices) as an initializer so that the following main
    # operator can access it.
    op_output_shape_last_dim = operator.get_output_shape()[-1]
    weights = np.array(op.get_weights()[0].T).reshape(op_output_shape_last_dim,
                                                      op.input_dim).transpose().flatten().tolist()
    embedding_tensor_name = container.add_initializer_by_name(scope, op.weights[0].name, onnx_proto.TensorProto.FLOAT,
                                                              [op.input_dim, op_output_shape_last_dim], weights)
    # Create a Gather operator to extract the latent representation of each index
    container.add_node('Gather', [embedding_tensor_name, cast_name], operator.output_full_names[0],
                       name=operator.full_name)
