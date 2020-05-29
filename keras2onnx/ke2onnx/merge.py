###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numpy as np
from ..proto import keras
from ..common.onnx_ops import apply_add, apply_mul, apply_sub
from ..common.onnx_ops import apply_mean, apply_max, OnnxOperatorBuilder

_merge_layer_handlers = {keras.layers.Add: apply_add, keras.layers.Multiply: apply_mul,
                         keras.layers.Subtract: apply_sub, keras.layers.Average: apply_mean,
                         keras.layers.Maximum: apply_max}


def convert_keras_merge_layer(scope, operator, container):
    op = operator.raw_operator
    if isinstance(op, keras.layers.Subtract) and len(operator.inputs) > 2:
        raise RuntimeError(
            'Expected two inputs but got %s. Their names are %s' % (len(operator.inputs), operator.input_full_names))

    apply_merge_operation = _merge_layer_handlers[type(op)]

    intermediate_tensor_name = None
    for i in range(len(operator.inputs) - 1):
        if i == 0:
            left_tensor_name = operator.inputs[0].full_name
            right_tensor_name = operator.inputs[1].full_name
        else:
            if intermediate_tensor_name is None:
                raise RuntimeError('Tensor name cannot be None')
            left_tensor_name = intermediate_tensor_name
            right_tensor_name = operator.inputs[i + 1].full_name

        if (len(operator.inputs) == 2 and i == 0) or (len(operator.inputs) > 2 and i == len(operator.inputs) - 2):
            # At the last iteration, we need to put the result to Keras layer's output tensor
            intermediate_tensor_name = operator.outputs[0].full_name
        else:
            # Keep accumulate changes through iterations using buffer tensors
            intermediate_tensor_name = scope.get_unique_variable_name('intermediate_tensor')
        apply_merge_operation(scope, [left_tensor_name, right_tensor_name], intermediate_tensor_name, container)

    if operator.output_masks:
        # Keras merge layer compute mask
        #    masks = [array_ops.expand_dims(m, axis=0) for m in mask if m is not None]
        #    return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)
        oopb = OnnxOperatorBuilder(container, scope)
        expanded = []
        for idx_, i_ in enumerate(operator.input_masks):
            expanded.append(oopb.add_node('Unsqueeze', i_.full_name, i_.full_name + '_i' + str(idx_), axes=[0]))

        if len(expanded) > 1:
            concat = oopb.apply_concat(expanded, name=operator.full_name + '_concat')
        else:
            concat = expanded[0]
        cast = oopb.add_node('Cast', concat, name=operator.full_name + '_cast', to=1)
        reduced = oopb.add_node('ReduceSum', cast, name=operator.full_name + '_reduced', op_version=1, axes=[0],
                                keepdims=0)
        oopb.apply_op_with_output('apply_greater',
                                  [reduced, np.array([0], dtype=np.float32)],
                                  [operator.output_masks[0].full_name],
                                  name=operator.raw_operator.name)
