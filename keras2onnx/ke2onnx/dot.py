###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import numbers
import numpy as np

from ..proto import is_keras_later_than
from ..common import cvtfunc
from ..common.onnx_ops import OnnxOperatorBuilder


# There is a breaking logic change for keras tensorflow_backend batch_dot after keras 2.2.4
# Assume input shape is (2, 3, 4, 12, 3) and (2, 3, 4, 15, 3), with axes 4
# For keras 2.2.4 and before, the output shape is (2, 3, 4, 12, 15)
# After that, the output shape is (2, 3, 4, 12, 3, 4, 15)
# See https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py for details.
def _calculate_keras_dot_output_shapes(operator):
    if not is_keras_later_than("2.2.4"):
        op = operator.raw_operator
        shape = []
        for i in op.output.shape:
            if isinstance(i.value, numbers.Integral):
                shape.append(i.value)
            else:
                shape.append(None)
        operator.outputs[0].type.shape = shape


def _preprocessing(op):
    if len(op.input_shape) > 2:
        raise RuntimeError('Unsupported number of input = %s > 2' % len(op.input_shape))
    x_shape = op.input_shape[0]
    y_shape = op.input_shape[1]
    x_shape = [x_ if x_ is not None else -1 for x_ in x_shape]
    y_shape = [y_ if y_ is not None else -1 for y_ in y_shape]
    x_shape = np.asarray(x_shape, dtype=np.int64)
    y_shape = np.asarray(y_shape, dtype=np.int64)

    x_ndim = len(x_shape)
    y_ndim = len(y_shape)
    x_batch_size = x_shape[0]
    y_batch_size = y_shape[0]

    if x_batch_size != y_batch_size:
        raise RuntimeError(
            'Can not do batch_dot on inputs with different batch sizes.' + str(x_shape) + ' and ' + str(y_shape))

    return x_ndim, y_ndim, x_shape, y_shape


def convert_keras_dot_224(scope, operator, container):
    op = operator.raw_operator
    x_ndim, y_ndim, _, _ = _preprocessing(op)
    oopb = OnnxOperatorBuilder(container, scope)

    axes = op.axes
    if isinstance(axes, int):
        axes = (axes, axes)
    if axes is None:
        # behaves like tf.batch_matmul as default
        axes = [x_ndim - 1, y_ndim - 2]
    if any([isinstance(a, (list, tuple)) for a in axes]):
        raise ValueError('Multiple target dimensions are not supported. ' +
                         'Expected: None, int, (int, int), ' +
                         'Provided: ' + str(axes))

    normalized_input_names = []
    if op.normalize:
        for i_, tensor_name in enumerate(operator.input_full_names):
            normalized_tensor_name = oopb.apply_normalization(tensor_name, name=operator.full_name+'_normalize', axis=axes[i_])
            normalized_input_names += normalized_tensor_name
    else:
        normalized_input_names = operator.input_full_names

    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y_shape_node = oopb.add_node('Shape',
                                     [normalized_input_names[1]],
                                     operator.inputs[0].full_name + '_y_shape')
        y_shape_concat = oopb.add_node('Concat',
                                       [y_shape_node,
                                        ('_ones', oopb.int64, np.array([1] * (diff), dtype='int64'))],
                                       operator.inputs[0].full_name + '_y_shape_concat',
                                       axis=0)
        y_reshape = oopb.add_node('Reshape',
                                  [normalized_input_names[1], y_shape_concat],
                                  operator.inputs[0].full_name + '_y_reshape')
        x_reshape = normalized_input_names[0]
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x_shape_node = oopb.add_node('Shape',
                                     [normalized_input_names[0]],
                                     operator.inputs[0].full_name + '_y_shape')
        x_shape_concat = oopb.add_node('Concat',
                                       [x_shape_node,
                                        ('_ones', oopb.int64, np.array([1] * (diff), dtype='int64'))],
                                       operator.inputs[0].full_name + '_x_shape_concat',
                                       axis=0)
        x_reshape = oopb.add_node('Reshape',
                                  [normalized_input_names[0], x_shape_concat],
                                  operator.inputs[0].full_name + '_x_reshape')
        y_reshape = normalized_input_names[1]
    else:
        diff = 0
        x_reshape = normalized_input_names[0]
        y_reshape = normalized_input_names[1]

    max_ndim = max([x_ndim, y_ndim])
    if x_ndim == 2 and y_ndim == 2:
        if axes[0] == axes[1]:
            result_mul = oopb.add_node('Mul',
                                       [x_reshape, y_reshape],
                                       operator.inputs[0].full_name + '_result_mul')
            out = oopb.add_node('ReduceSum',
                                [result_mul],
                                operator.inputs[0].full_name + '_out',
                                axes=[axes[0]])
        else:
            x_transpose = oopb.add_node('Transpose',
                                        [x_reshape],
                                        operator.inputs[0].full_name + '_x_transpose',
                                        perm=[1, 0])
            result_mul = oopb.add_node('Mul',
                                       [x_transpose, y_reshape],
                                       operator.inputs[0].full_name + '_result_mul')
            out = oopb.add_node('ReduceSum',
                                [result_mul],
                                operator.inputs[0].full_name + '_out',
                                axes=[axes[1]])
    else:
        if axes is not None:
            adj_x = None if axes[0] == max_ndim - 1 else True
            adj_y = True if axes[1] == max_ndim - 1 else None
        else:
            adj_x = None
            adj_y = None

        transpose_perm = list(range(max_ndim))
        temp = transpose_perm[-1]
        transpose_perm[-1] = transpose_perm[-2]
        transpose_perm[-2] = temp

        if adj_x:
            x_transpose_2 = oopb.add_node('Transpose',
                                          [x_reshape],
                                          operator.inputs[0].full_name + '_x_transpose_2',
                                          perm=transpose_perm)
        else:
            x_transpose_2 = x_reshape
        if adj_y:
            y_transpose_2 = oopb.add_node('Transpose',
                                          [y_reshape],
                                          operator.inputs[0].full_name + '_y_transpose_2',
                                          perm=transpose_perm)
        else:
            y_transpose_2 = y_reshape
        out = oopb.add_node('MatMul',
                            [x_transpose_2, y_transpose_2],
                            operator.inputs[0].full_name + '_out')
    matrix_len = max_ndim
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out_squeeze = oopb.add_node('Squeeze',
                                    [out],
                                    operator.inputs[0].full_name + '_out_squeeze',
                                    axes=list(range(idx, idx + diff)))
        matrix_len = matrix_len - diff
    else:
        out_squeeze = out

    if matrix_len == 1:
        out_expand = oopb.add_node('Unsqueeze',
                                   [out_squeeze],
                                   operator.inputs[0].full_name + '_out_expand',
                                   axes=[1])
    else:
        out_expand = out_squeeze
    container.add_node('Identity', out_expand, operator.output_full_names,
                       name=scope.get_unique_operator_name('Identity'))


def convert_keras_dot_post_224(scope, operator, container):
    op = operator.raw_operator
    x_ndim, y_ndim, x_shape, y_shape = _preprocessing(op)
    oopb = OnnxOperatorBuilder(container, scope)

    orig_x_ndim = x_ndim
    orig_y_ndim = y_ndim

    if isinstance(op.axes, int):
        if op.axes < 0:
            axes = [op.axes % len(x_shape), op.axes % len(y_shape)]
        else:
            axes = [op.axes] * 2
    else:
        axes = op.axes
    a0, a1 = axes

    normalized_input_names = []
    if op.normalize:
        for i_, tensor_name in enumerate(operator.input_full_names):
            normalized_tensor_name = oopb.apply_normalization(tensor_name, name=operator.full_name, axis=axes[i_])
            normalized_input_names += normalized_tensor_name
    else:
        normalized_input_names = operator.input_full_names

    if x_shape[a0] != y_shape[a1]:
        raise RuntimeError('Dimension incompatibility: %s != %s' % (x_shape[axes[0]], y_shape[axes[1]]))

    if x_ndim == 2:
        x_expand = oopb.add_node('Unsqueeze',
                                 [normalized_input_names[0]],
                                 operator.inputs[0].full_name + '_expand',
                                 axes=[1])
        a0 += 1
        x_ndim += 1
    else:
        x_expand = normalized_input_names[0]

    if y_ndim == 2:
        y_expand = oopb.add_node('Unsqueeze',
                                 [normalized_input_names[1]],
                                 operator.inputs[1].full_name + '_expand',
                                 axes=[2])
        y_ndim += 1
    else:
        y_expand = normalized_input_names[1]

    # bring x's dimension to be reduced to last axis.
    if a0 != x_ndim - 1:
        pattern = list(range(x_ndim))
        for i in range(a0, x_ndim - 1):
            pattern[i] = pattern[i + 1]
        pattern[-1] = a0
        x_transpose = oopb.add_node('Transpose',
                                    [x_expand],
                                    operator.inputs[0].full_name + '_transpose',
                                    perm=pattern)
    else:
        x_transpose = x_expand

    # bring y's dimension to be reduced to axis 1.
    if a1 != 1:
        pattern = list(range(y_ndim))
        for i in range(a1, 1, -1):
            pattern[i] = pattern[i - 1]
        pattern[1] = a1
        y_transpose = oopb.add_node('Transpose',
                                    [y_expand],
                                    operator.inputs[1].full_name + '_transpose',
                                    perm=pattern)
    else:
        y_transpose = y_expand

    # normalize both inputs to rank 3.
    if x_ndim > 3:
        # squash middle dimensions of x.
        x_shape_node = oopb.add_node('Shape',
                                     [x_transpose],
                                     operator.inputs[0].full_name + '_x_shape')
        x_mid_dims = oopb.add_node('Slice',
                                   [x_shape_node,
                                    ('_start', oopb.int64, np.array([1], dtype='int64')),
                                    ('_end', oopb.int64, np.array([-1], dtype='int64')),
                                    ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                    ],
                                   operator.inputs[0].full_name + '_x_mid_dims')
        x_mid_dims_cast = oopb.add_node('Cast',
                                        [x_mid_dims],
                                        operator.inputs[0].full_name + '_x_mid_dims_cast',
                                        to=6)
        x_squashed_dim = oopb.add_node('ReduceProd',
                                       [x_mid_dims_cast],
                                       operator.inputs[0].full_name + '_x_squashed_dim')
        x_squashed_dim_cast = oopb.add_node('Cast',
                                            [x_squashed_dim],
                                            operator.inputs[0].full_name + '_x_squashed_dim_cast',
                                            to=7)
        x_shape_0 = oopb.add_node('Slice',
                                  [x_shape_node,
                                   ('_start', oopb.int64, np.array([0], dtype='int64')),
                                   ('_end', oopb.int64, np.array([1], dtype='int64')),
                                   ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                   ],
                                  operator.inputs[0].full_name + '_x_shape_0')
        x_shape_1 = oopb.add_node('Slice',
                                  [x_shape,
                                   ('_start', oopb.int64, np.array([-1], dtype='int64')),
                                   ('_end', oopb.int64, np.array([np.iinfo(np.int64).max], dtype='int64')),
                                   ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                   ],
                                  operator.inputs[0].full_name + '_x_shape_1')
        x_squashed_shape = oopb.add_node('Concat',
                                         [x_shape_0, x_squashed_dim_cast, x_shape_1],
                                         operator.inputs[0].full_name + '_x_squashed_shape',
                                         axis=0)
        x_reshape = oopb.add_node('Reshape',
                                  [x_transpose, x_squashed_shape],
                                  operator.inputs[0].full_name + '_x_reshape')
        x_squashed = True
    else:
        x_reshape = x_transpose
        x_squashed = False

    if y_ndim > 3:
        # squash trailing dimensions of y.
        y_shape_node = oopb.add_node('Shape',
                                     [y_transpose],
                                     operator.inputs[0].full_name + '_y_shape')
        y_trail_dims = oopb.add_node('Slice',
                                     [y_shape_node,
                                      ('_start', oopb.int64, np.array([2], dtype='int64')),
                                      ('_end', oopb.int64, np.array([np.iinfo(np.int64).max], dtype='int64')),
                                      ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                      ],
                                     operator.inputs[0].full_name + '_y_trail_dims')
        y_trail_dims_cast = oopb.add_node('Cast',
                                          [y_trail_dims],
                                          operator.inputs[0].full_name + '_y_trail_dims_cast',
                                          to=6)
        y_squashed_dim = oopb.add_node('ReduceProd',
                                       [y_trail_dims_cast],
                                       operator.inputs[0].full_name + '_y_squashed_dim')
        y_squashed_dim_cast = oopb.add_node('Cast',
                                            [y_squashed_dim],
                                            operator.inputs[0].full_name + '_y_squashed_dim_cast',
                                            to=7)
        y_shape_0 = oopb.add_node('Slice',
                                  [y_shape_node,
                                   ('_start', oopb.int64, np.array([0], dtype='int64')),
                                   ('_end', oopb.int64, np.array([1], dtype='int64')),
                                   ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                   ],
                                  operator.inputs[0].full_name + '_y_shape_0')
        y_shape_1 = oopb.add_node('Slice',
                                  [y_shape_node,
                                   ('_start', oopb.int64, np.array([1], dtype='int64')),
                                   ('_end', oopb.int64, np.array([2], dtype='int64')),
                                   ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                   ],
                                  operator.inputs[0].full_name + '_y_shape_1')
        y_squashed_shape = oopb.add_node('Concat',
                                         [y_shape_0, y_shape_1, y_squashed_dim_cast],
                                         operator.inputs[0].full_name + '_y_squashed_shape',
                                         axis=0)
        y_reshape = oopb.add_node('Reshape',
                                  [y_transpose, y_squashed_shape],
                                  operator.inputs[0].full_name + '_y_reshape')

        y_squashed = True
    else:
        y_reshape = y_transpose
        y_squashed = False

    matmul = oopb.add_node('MatMul',
                           [x_reshape, y_reshape],
                           operator.inputs[0].full_name + '_matmul')

    # if inputs were squashed, we have to reshape the matmul output.
    if x_squashed or y_squashed:
        output_shape = oopb.add_node('Shape',
                                     [matmul],
                                     operator.inputs[0].full_name + '_output_shape')
    else:
        output_shape = matmul
    do_reshape = False

    if x_squashed:
        output_shape_x_0 = oopb.add_node('Slice',
                                         [output_shape,
                                          ('_start', oopb.int64, np.array([0], dtype='int64')),
                                          ('_end', oopb.int64, np.array([1], dtype='int64')),
                                          ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                          ],
                                         operator.inputs[0].full_name + '_output_shape_x_0')
        output_shape_x_1 = oopb.add_node('Slice',
                                         [output_shape,
                                          ('_start', oopb.int64, np.array([-1], dtype='int64')),
                                          ('_end', oopb.int64, np.array([np.iinfo(np.int64).max], dtype='int64')),
                                          ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                          ],
                                         operator.inputs[0].full_name + '_output_shape_x_1')
        output_shape_x = oopb.add_node('Concat',
                                       [output_shape_x_0, x_mid_dims, output_shape_x_1],
                                       operator.inputs[0].full_name + '_output_shape_x',
                                       axis=0)
        do_reshape = True
    else:
        output_shape_x = output_shape

    if y_squashed:
        output_shape_y_0 = oopb.add_node('Slice',
                                         [output_shape_x,
                                          ('_start', oopb.int64, np.array([0], dtype='int64')),
                                          ('_end', oopb.int64, np.array([-1], dtype='int64')),
                                          ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                          ],
                                         operator.inputs[0].full_name + '_output_shape_y_0')
        output_shape_y = oopb.add_node('Concat',
                                       [output_shape_y_0, y_trail_dims],
                                       operator.inputs[0].full_name + '_output_shape_y',
                                       axis=0)
        do_reshape = True
    else:
        output_shape_y = output_shape_x

    if do_reshape:
        output_reshape = oopb.add_node('Reshape',
                                       [matmul, output_shape_y],
                                       operator.inputs[0].full_name + '_output_reshape')
    else:
        output_reshape = matmul

    # if the inputs were originally rank 2, we remove the added 1 dim.
    if orig_x_ndim == 2:
        container.add_node('Squeeze', output_reshape, operator.output_full_names,
                           name=scope.get_unique_operator_name('Squeeze'), axes=[1])
    elif orig_y_ndim == 2:
        container.add_node('Squeeze', output_reshape, operator.output_full_names,
                           name=scope.get_unique_operator_name('Squeeze'), axes=[y_ndim - 1])
    else:
        container.add_node('Identity', output_reshape, operator.output_full_names,
                           name=scope.get_unique_operator_name('Identity'))


@cvtfunc(shape_infer=_calculate_keras_dot_output_shapes)
def convert_keras_dot(scope, operator, container):
    from keras2onnx.proto import keras
    if not is_keras_later_than("2.2.4"):
        print('dot_224')
        print('keras=' + keras.__version__.split('-')[0])
        convert_keras_dot_224(scope, operator, container)
    else:
        print('dot_post_224')
        print('keras=' + keras.__version__.split('-')[0])
        convert_keras_dot_post_224(scope, operator, container)
