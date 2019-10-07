###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import sys
import onnx
import numpy as np
import keras2onnx
from keras2onnx.proto import keras, is_keras_older_than
from keras2onnx.common.onnx_ops import apply_identity
from onnx import onnx_pb, helper

working_path = os.path.abspath(os.path.dirname(__file__))
tmp_path = os.path.join(working_path, 'temp')


def print_mismatches(case_name, list_idx, expected_list, actual_list, rtol=1.e-3, atol=1.e-6):
    diff_list = abs(expected_list - actual_list)
    count_total = len(expected_list)
    count_error = 0

    for e_, a_, d_ in zip(expected_list, actual_list, diff_list):
        if d_ > atol + rtol * abs(a_):
            if count_error < 10:  # print the first 10 mismatches
                print(
                    "case = " + case_name + ", result mismatch for expected = " + str(e_) +
                    ", actual = " + str(a_), file=sys.stderr)
            count_error = count_error + 1

    print("case = " + case_name + ", " +
          str(count_error) + " mismatches out of " + str(count_total) + " for list " + str(list_idx),
          file=sys.stderr)


def run_onnx_runtime(case_name, onnx_model, data, expected, model_files, rtol=1.e-3, atol=1.e-6):
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    temp_model_file = os.path.join(tmp_path, 'temp_' + case_name + '.onnx')
    onnx.save_model(onnx_model, temp_model_file)
    try:
        import onnxruntime
        sess = onnxruntime.InferenceSession(temp_model_file)
    except ImportError:
        return True

    if isinstance(data, dict):
        feed_input = data
    else:
        data = data if isinstance(data, list) else [data]
        input_names = sess.get_inputs()
        # to avoid too complicated test code, we restrict the input name in Keras test cases must be
        # in alphabetical order. It's always true unless there is any trick preventing that.
        feed = zip(sorted(i_.name for i_ in input_names), data)
        feed_input = dict(feed)
    actual = sess.run(None, feed_input)

    if expected is None:
        return

    if not isinstance(expected, list):
        expected = [expected]

    res = all(np.allclose(expected[n_], actual[n_], rtol=rtol, atol=atol) for n_ in range(len(expected)))

    if res and temp_model_file not in model_files:  # still keep the failed case files for the diagnosis.
        model_files.append(temp_model_file)

    if not res:
        for n_ in range(len(expected)):
            expected_list = expected[n_].flatten()
            actual_list = actual[n_].flatten()
            print_mismatches(case_name, n_, expected_list, actual_list, rtol, atol)

    return res


def run_image(model, model_files, img_path, model_name='onnx_conversion', rtol=1.e-3, atol=1.e-5, color_mode="rgb",
              target_size=224):
    preprocess_input = keras.applications.resnet50.preprocess_input
    image = keras.preprocessing.image

    try:
        if not isinstance(target_size, tuple):
            target_size = (target_size, target_size)
        if is_keras_older_than("2.2.3"):
            # color_mode is not supported in old keras version
            img = image.load_img(img_path, target_size=target_size)
        else:
            img = image.load_img(img_path, color_mode=color_mode, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        if color_mode == "rgb":
            x = preprocess_input(x)
    except FileNotFoundError:
        return False, 'The image data does not exist.'

    msg = ''
    preds = None
    try:
        preds = model.predict(x)
    except RuntimeError:
        msg = 'keras prediction throws an exception for model ' + model.name + ', skip comparison.'

    onnx_model = keras2onnx.convert_keras(model, model.name)
    res = run_onnx_runtime(model_name, onnx_model, x, preds, model_files, rtol=rtol, atol=atol)
    return res, msg


tf2onnx = keras2onnx.wrapper.tf2onnx


# This is for Pad opset 11 which is now a contrib op, TODO: need onnx schema update for Pad
def on_Pad(ctx, node, name, args):
    node.type = "Pad"
    node.domain = 'com.microsoft'
    mode = node.get_attr("mode")
    if mode:
        mode = mode.s.decode("utf-8").lower()
        node.set_attr("mode", mode)
    if mode not in [None, "constant", "reflect"]:
        raise ValueError(mode + " pad mode is not supported")

    origin_dtype = ctx.get_dtype(node.output[0])
    cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[1])
    cast_node.set_attr("to", onnx_pb.TensorProto.INT64)
    ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT64)
    ctx.copy_shape(node.name, cast_node.output[0])

    attrs = {'perm': [1, 0]}
    transpose_node = ctx.make_node("Transpose", [cast_node.output[0]], name=tf2onnx.utils.make_name(node.name),
                                   attr=attrs)

    const_name = tf2onnx.utils.make_name(node.name)

    const_array = ctx.make_const(const_name, np.array([-1], dtype=np.int64))

    reshape = ctx.make_node("Reshape", [transpose_node.output[0], const_array.output[0]])
    ctx.replace_input(node, node.input[1], reshape.output[0])

    if origin_dtype not in [onnx_pb.TensorProto.FLOAT16, onnx_pb.TensorProto.FLOAT,
                            onnx_pb.TensorProto.DOUBLE]:
        cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[0])
        cast_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
        ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
        ctx.copy_shape(node.name, cast_node.output[0])

        cast_back_node = ctx.insert_new_node_on_output("Cast", node.output[0],
                                                       name=tf2onnx.utils.make_name(node.name) + "_castback")
        cast_back_node.set_attr("to", origin_dtype)
        ctx.set_dtype(cast_back_node.output[0], origin_dtype)
        ctx.copy_shape(node.name, cast_back_node.output[0])


def on_CropAndResize(ctx, node, name, args):
    node.type = "CropAndResize"
    node.domain = 'com.microsoft'
    mode = node.get_attr("method")
    if mode:
        mode_value = helper.get_attribute_value(mode)
        del node.attr['method']
        node.set_attr("mode", mode_value)

    transpose_node = ctx.insert_new_node_on_input(node, "Transpose", node.input[0])
    transpose_node.set_attr("perm", [0, 3, 1, 2])
    ctx.set_dtype(transpose_node.output[0], onnx_pb.TensorProto.INT64)

    transpose_node_2 = ctx.insert_new_node_on_output("Transpose", node.output[0],
                                                     name=tf2onnx.utils.make_name(node.name) + "_transpose_final")
    transpose_node_2.set_attr("perm", [0, 2, 3, 1])
    ctx.set_dtype(transpose_node_2.output[0], onnx_pb.TensorProto.INT64)


def on_GatherNd(ctx, node, name, args):
    node.type = "GatherND"
    node.domain = "com.microsoft"


# convert keras_contrib.layers.InstanceNormalization
def convert_InstanceNormalizationLayer(scope, operator, container):
    from keras2onnx.common.onnx_ops import OnnxOperatorBuilder
    op = operator.raw_operator
    params = op.get_weights()
    assert len(op.input_shape) == 4
    beta = params[0].reshape(1, 1, 1, 1).astype('float32')
    gamma = params[1].reshape(1, 1, 1, 1).astype('float32')
    oopb = OnnxOperatorBuilder(container, scope)

    reducemean_1 = oopb.add_node('ReduceMean',
                                 [operator.inputs[0].full_name],
                                 operator.inputs[0].full_name + '_reduce_mean_1',
                                 axes=[1, 2, 3], keepdims=1)

    sub_1 = oopb.add_node('Sub',
                          [operator.inputs[0].full_name, reducemean_1],
                          operator.inputs[0].full_name + '_sub_1')

    mul = oopb.add_node('Mul',
                        [sub_1, sub_1],
                        operator.inputs[0].full_name + '_mul')

    reducemean_2 = oopb.add_node('ReduceMean',
                                 [mul],
                                 operator.inputs[0].full_name + '_reduce_mean_2',
                                 axes=[1, 2, 3], keepdims=1)

    sqrt = oopb.add_node('Sqrt',
                         [reducemean_2],
                         operator.inputs[0].full_name + '_sqrt')

    add = oopb.add_node('Add',
                        [sqrt,
                         ('_start', oopb.float, np.array([op.epsilon], dtype='float32'))],
                        operator.inputs[0].full_name + '_add')

    div = oopb.add_node('Div',
                        [sub_1, add],
                        operator.inputs[0].full_name + '_div')

    mul_scale = oopb.add_node('Mul',
                              [div,
                               ('_start', oopb.float, beta)],
                              operator.inputs[0].full_name + '_mul_scale')

    add_bias = oopb.add_node('Add',
                             [mul_scale,
                              ('_start', oopb.float, gamma)],
                             operator.inputs[0].full_name + '_add_bias')

    apply_identity(scope, add_bias, operator.outputs[0].full_name, container)


tf2onnx_contrib_op_conversion = {
    'GatherNd': (on_GatherNd, []),
    'CropAndResize': (on_CropAndResize, []),
    'Pad': (on_Pad, []),
    'PadV2': (on_Pad, [])
}
