import os
import sys
import numpy as np
import skimage
import onnx
import keras2onnx

from mrcnn.config import Config
from mrcnn.model import BatchNorm, DetectionLayer
from mrcnn import model as modellib
from mrcnn import visualize

from keras2onnx import set_converter
from keras2onnx.ke2onnx.batch_norm import convert_keras_batch_normalization
from keras2onnx.proto import onnx_proto
from keras2onnx.common.onnx_ops import apply_transpose, apply_identity
from keras2onnx.common.onnx_ops import OnnxOperatorBuilder
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import tf2onnx_contrib_op_conversion


ROOT_DIR = os.path.abspath("./")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


def convert_BatchNorm(scope, operator, container):
    convert_keras_batch_normalization(scope, operator, container)


def convert_apply_box_deltas_graph(scope, operator, container, oopb, box_transpose, score_identity, deltas_transpose, windows_transpose):
    box_squeeze = scope.get_unique_variable_name('box_squeeze')
    attrs = {'axes': [0]}
    container.add_node('Squeeze', box_transpose, box_squeeze, op_version=operator.target_opset,
                       **attrs)
    # output shape: [spatial_dimension, 4]

    deltas_squeeze = scope.get_unique_variable_name('deltas_squeeze')
    attrs = {'axes': [0]}
    container.add_node('Squeeze', deltas_transpose, deltas_squeeze, op_version=operator.target_opset,
                       **attrs)
    # output shape: [spatial_dimension, num_classes, 4]

    score_squeeze = scope.get_unique_variable_name('score_squeeze')
    attrs = {'axes': [0]}
    container.add_node('Squeeze', score_identity, score_squeeze, op_version=operator.target_opset,
                       **attrs)
    # output shape: [spatial_dimension, num_classes]

    class_ids = scope.get_unique_variable_name('class_ids')
    attrs = {'axis': 1}
    container.add_node('ArgMax', score_squeeze, class_ids, op_version=operator.target_opset,
                       **attrs)
    # output shape: [spatial_dimension, 1]

    prob_shape = oopb.add_node('Shape',
                                 [score_squeeze],
                                 operator.inputs[1].full_name + '_prob_shape')
    prob_shape_0 = oopb.add_node('Slice',
                         [prob_shape,
                          ('_start', oopb.int64, np.array([0], dtype='int64')),
                          ('_end', oopb.int64, np.array([1], dtype='int64')),
                          ('_axes', oopb.int64, np.array([0], dtype='int64'))
                          ],
                         operator.inputs[1].full_name + '_prob_shape_0')
    prob_range = oopb.add_node('Range',
                         [('_start', oopb.int64, np.array([0], dtype='int64')),
                          prob_shape_0,
                          # ('_limit', oopb.int64, np.array([1000], dtype='int64')),
                          ('_delta', oopb.int64, np.array([1], dtype='int64'))
                          ],
                         operator.inputs[1].full_name + '_prob_range',
                         op_domain='com.microsoft')

    attrs = {'axes': [1]}
    prob_range_unsqueeze = oopb.add_node('Unsqueeze',
                         [prob_range],
                         operator.inputs[1].full_name + '_prob_range_unsqueeze',
                         **attrs)
    # output shape: [spatial_dimension, 1]

    attrs = {'axis': 1}
    indices = oopb.add_node('Concat',
                         [prob_range_unsqueeze,
                          class_ids
                          ],
                         operator.inputs[1].full_name + '_indices', **attrs)
    # output shape: [spatial_dimension, 2]

    deltas_specific = oopb.add_node('GatherND',
                         [deltas_squeeze, indices],
                         operator.inputs[2].full_name + '_deltas_specific',
                         op_domain='com.microsoft')
    # output shape: [spatial_dimension, 4]

    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2], dtype='float32')
    delta_mul_output = oopb.add_node('Mul',
                                     [deltas_specific,
                                      ('_mul_constant', oopb.float, BBOX_STD_DEV)
                                     ],
                                     operator.inputs[2].full_name + '_mul')
    # output shape: [spatial_dimension, 4]

    box_0 = oopb.add_node('Slice',
                         [box_squeeze,
                          ('_start', oopb.int64, np.array([0], dtype='int64')),
                          ('_end', oopb.int64, np.array([1], dtype='int64')),
                          ('_axes', oopb.int64, np.array([1], dtype='int64'))
                          ],
                         operator.inputs[0].full_name + '_sliced_0')
    box_1 = oopb.add_node('Slice',
                          [box_squeeze,
                           ('_start', oopb.int64, np.array([1], dtype='int64')),
                           ('_end', oopb.int64, np.array([2], dtype='int64')),
                           ('_axes', oopb.int64, np.array([1], dtype='int64'))
                           ],
                          operator.inputs[0].full_name + '_sliced_1')
    box_2 = oopb.add_node('Slice',
                          [box_squeeze,
                           ('_start', oopb.int64, np.array([2], dtype='int64')),
                           ('_end', oopb.int64, np.array([3], dtype='int64')),
                           ('_axes', oopb.int64, np.array([1], dtype='int64'))
                           ],
                          operator.inputs[0].full_name + '_sliced_2')
    box_3 = oopb.add_node('Slice',
                          [box_squeeze,
                           ('_start', oopb.int64, np.array([3], dtype='int64')),
                           ('_end', oopb.int64, np.array([4], dtype='int64')),
                           ('_axes', oopb.int64, np.array([1], dtype='int64'))
                           ],
                          operator.inputs[0].full_name + '_sliced_3')

    delta_0 = oopb.add_node('Slice',
                         [delta_mul_output,
                          ('_start', oopb.int64, np.array([0], dtype='int64')),
                          ('_end', oopb.int64, np.array([1], dtype='int64')),
                          ('_axes', oopb.int64, np.array([1], dtype='int64'))
                          ],
                         operator.inputs[3].full_name + '_sliced_0')
    delta_1 = oopb.add_node('Slice',
                          [delta_mul_output,
                           ('_start', oopb.int64, np.array([1], dtype='int64')),
                           ('_end', oopb.int64, np.array([2], dtype='int64')),
                           ('_axes', oopb.int64, np.array([1], dtype='int64'))
                           ],
                          operator.inputs[3].full_name + '_sliced_1')
    delta_2 = oopb.add_node('Slice',
                          [delta_mul_output,
                           ('_start', oopb.int64, np.array([2], dtype='int64')),
                           ('_end', oopb.int64, np.array([3], dtype='int64')),
                           ('_axes', oopb.int64, np.array([1], dtype='int64'))
                           ],
                          operator.inputs[3].full_name + '_sliced_2')
    delta_3 = oopb.add_node('Slice',
                          [delta_mul_output,
                           ('_start', oopb.int64, np.array([3], dtype='int64')),
                           ('_end', oopb.int64, np.array([4], dtype='int64')),
                           ('_axes', oopb.int64, np.array([1], dtype='int64'))
                           ],
                          operator.inputs[3].full_name + '_sliced_3')

    height = oopb.add_node('Sub',
                          [box_2, box_0],
                          operator.inputs[0].full_name + '_height')
    width = oopb.add_node('Sub',
                          [box_3, box_1],
                          operator.inputs[0].full_name + '_width')

    half_height_0 = oopb.add_node('Mul',
                                  [height,
                                   ('_mul_constant', oopb.float, np.array([0.5], dtype='float32'))
                                  ],
                                  operator.inputs[0].full_name + '_half_height_0')
    half_width_0 = oopb.add_node('Mul',
                                  [width,
                                   ('_mul_constant', oopb.float, np.array([0.5], dtype='float32'))
                                  ],
                                  operator.inputs[0].full_name + '_half_width_0')
    center_y_0 = oopb.add_node('Add',
                               [box_0, half_height_0],
                               operator.inputs[0].full_name + '_center_y_0')
    center_x_0 = oopb.add_node('Add',
                               [box_1, half_width_0],
                               operator.inputs[0].full_name + '_center_x_0')

    delta_height = oopb.add_node('Mul',
                               [delta_0, height],
                               operator.inputs[0].full_name + '_delta_height')
    delta_width = oopb.add_node('Mul',
                               [delta_1, width],
                               operator.inputs[0].full_name + '_delta_width')
    center_y_1 = oopb.add_node('Add',
                               [center_y_0, delta_height],
                               operator.inputs[0].full_name + '_center_y_1')
    center_x_1 = oopb.add_node('Add',
                               [center_x_0, delta_width],
                               operator.inputs[0].full_name + '_center_x_1')

    delta_2_exp = oopb.add_node('Exp',
                                [delta_2],
                                operator.inputs[0].full_name + '_delta_2_exp')
    delta_3_exp = oopb.add_node('Exp',
                                [delta_3],
                                operator.inputs[0].full_name + '_delta_3_exp')
    height_exp = oopb.add_node('Mul',
                                 [height, delta_2_exp],
                                 operator.inputs[0].full_name + '_height_exp')
    width_exp = oopb.add_node('Mul',
                                [width, delta_3_exp],
                                operator.inputs[0].full_name + '_width_exp')

    half_height_1 = oopb.add_node('Mul',
                                  [height_exp,
                                   ('_mul_constant', oopb.float, np.array([0.5], dtype='float32'))
                                  ],
                                  operator.inputs[0].full_name + '_half_height_1')
    half_width_1 = oopb.add_node('Mul',
                                  [width_exp,
                                   ('_mul_constant', oopb.float, np.array([0.5], dtype='float32'))
                                  ],
                                  operator.inputs[0].full_name + '_half_width_1')
    y1 = oopb.add_node('Sub',
                          [center_y_1, half_height_1],
                          operator.inputs[0].full_name + '_y1')
    x1 = oopb.add_node('Sub',
                          [center_x_1, half_width_1],
                          operator.inputs[0].full_name + '_x1')
    y2 = oopb.add_node('Add',
                               [y1, height_exp],
                               operator.inputs[0].full_name + '_y2')
    x2 = oopb.add_node('Add',
                               [x1, width_exp],
                               operator.inputs[0].full_name + '_x2')

    windows_squeeze = scope.get_unique_variable_name('windows_squeeze')
    attrs = {'axes': [0]}
    container.add_node('Squeeze', windows_transpose, windows_squeeze, op_version=operator.target_opset,
                       **attrs)
    wy1 = oopb.add_node('Slice',
                         [windows_squeeze,
                          ('_start', oopb.int64, np.array([0], dtype='int64')),
                          ('_end', oopb.int64, np.array([1], dtype='int64')),
                          ('_axes', oopb.int64, np.array([0], dtype='int64'))
                          ],
                         operator.inputs[0].full_name + '_windows_0')
    wx1 = oopb.add_node('Slice',
                          [windows_squeeze,
                           ('_start', oopb.int64, np.array([1], dtype='int64')),
                           ('_end', oopb.int64, np.array([2], dtype='int64')),
                           ('_axes', oopb.int64, np.array([0], dtype='int64'))
                           ],
                          operator.inputs[0].full_name + '_windows_1')
    wy2 = oopb.add_node('Slice',
                          [windows_squeeze,
                           ('_start', oopb.int64, np.array([2], dtype='int64')),
                           ('_end', oopb.int64, np.array([3], dtype='int64')),
                           ('_axes', oopb.int64, np.array([0], dtype='int64'))
                           ],
                          operator.inputs[0].full_name + '_windows_2')
    wx2 = oopb.add_node('Slice',
                          [windows_squeeze,
                           ('_start', oopb.int64, np.array([3], dtype='int64')),
                           ('_end', oopb.int64, np.array([4], dtype='int64')),
                           ('_axes', oopb.int64, np.array([0], dtype='int64'))
                           ],
                          operator.inputs[0].full_name + '_windows_3')
    y1_min = oopb.add_node('Min',
                       [y1, wy2],
                       operator.inputs[0].full_name + '_y1_min')
    x1_min = oopb.add_node('Min',
                       [x1, wx2],
                       operator.inputs[0].full_name + '_x1_min')
    y2_min = oopb.add_node('Min',
                       [y2, wy2],
                       operator.inputs[0].full_name + '_y2_min')
    x2_min = oopb.add_node('Min',
                       [x2, wx2],
                       operator.inputs[0].full_name + '_x2_min')
    y1_max = oopb.add_node('Max',
                           [y1_min, wy1],
                           operator.inputs[0].full_name + '_y1_max')
    x1_max = oopb.add_node('Max',
                           [x1_min, wx1],
                           operator.inputs[0].full_name + '_x1_max')
    y2_max = oopb.add_node('Max',
                           [y2_min, wy1],
                           operator.inputs[0].full_name + '_y2_max')
    x2_max = oopb.add_node('Max',
                           [x2_min, wx1],
                           operator.inputs[0].full_name + '_x2_max')
    concat_result = scope.get_unique_variable_name(operator.output_full_names[0] + '_concat_result')
    attrs = {'axis': 1}
    container.add_node("Concat",
                       [y1_max, x1_max, y2_max, x2_max],
                       concat_result,
                       op_version=operator.target_opset,
                       name=operator.outputs[0].full_name + '_concat_result', **attrs)

    concat_unsqueeze = scope.get_unique_variable_name('concat_unsqueeze')
    attrs = {'axes': [0]}
    container.add_node('Unsqueeze', concat_result, concat_unsqueeze, op_version=operator.target_opset,
                       **attrs)
    return concat_unsqueeze


def norm_boxes_graph(scope, operator, container, oopb, image_meta):
    image_shapes = oopb.add_node('Slice',
                         [image_meta,
                          ('_start', oopb.int64, np.array([4], dtype='int64')),
                          ('_end', oopb.int64, np.array([7], dtype='int64')),
                          ('_axes', oopb.int64, np.array([1], dtype='int64'))
                          ],
                         operator.inputs[0].full_name + '_image_shapes')
    image_shape = oopb.add_node('Slice',
                                 [image_shapes,
                                  ('_start', oopb.int64, np.array([0], dtype='int64')),
                                  ('_end', oopb.int64, np.array([1], dtype='int64')),
                                  ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                  ],
                                 operator.inputs[0].full_name + '_image_shape')
    image_shape_squeeze = scope.get_unique_variable_name('image_shape_squeeze')
    attrs = {'axes': [0]}
    container.add_node('Squeeze', image_shape, image_shape_squeeze, op_version=operator.target_opset,
                       **attrs)
    window = oopb.add_node('Slice',
                            [image_meta,
                             ('_start', oopb.int64, np.array([7], dtype='int64')),
                             ('_end', oopb.int64, np.array([11], dtype='int64')),
                             ('_axes', oopb.int64, np.array([1], dtype='int64'))
                             ],
                            operator.inputs[0].full_name + '_window')
    h_norm = oopb.add_node('Slice',
                         [image_shape_squeeze,
                          ('_start', oopb.int64, np.array([0], dtype='int64')),
                          ('_end', oopb.int64, np.array([1], dtype='int64')),
                          ('_axes', oopb.int64, np.array([0], dtype='int64'))
                          ],
                         operator.inputs[0].full_name + '_h_norm')
    w_norm = oopb.add_node('Slice',
                           [image_shape_squeeze,
                            ('_start', oopb.int64, np.array([1], dtype='int64')),
                            ('_end', oopb.int64, np.array([2], dtype='int64')),
                            ('_axes', oopb.int64, np.array([0], dtype='int64'))
                            ],
                           operator.inputs[0].full_name + '_w_norm')
    h_norm_float = scope.get_unique_variable_name('h_norm_float')
    attrs = {'to': 1}
    container.add_node('Cast', h_norm, h_norm_float, op_version=operator.target_opset,
                       **attrs)
    w_norm_float = scope.get_unique_variable_name('w_norm_float')
    attrs = {'to': 1}
    container.add_node('Cast', w_norm, w_norm_float, op_version=operator.target_opset,
                       **attrs)
    hw_concat = scope.get_unique_variable_name(operator.inputs[0].full_name + '_hw_concat')
    attrs = {'axis': -1}
    container.add_node("Concat",
                       [h_norm_float, w_norm_float, h_norm_float, w_norm_float],
                       hw_concat,
                       op_version=operator.target_opset,
                       name=operator.inputs[0].full_name + '_hw_concat', **attrs)
    scale = oopb.add_node('Sub',
                          [hw_concat,
                           ('_sub', oopb.float, np.array([1.0], dtype='float32'))
                           ],
                          operator.inputs[0].full_name + '_scale')
    boxes_shift = oopb.add_node('Sub',
                          [window,
                           ('_sub', oopb.float, np.array([0.0, 0.0, 1.0, 1.0], dtype='float32'))
                           ],
                          operator.inputs[0].full_name + '_boxes_shift')
    divide = oopb.add_node('Div',
                            [boxes_shift, scale],
                            operator.inputs[0].full_name + '_divide')
    # output shape: [batch, 4]
    return divide


def convert_DetectionLayer(scope, operator, container):
    # type: (keras2onnx.common.InterimContext, keras2onnx.common.Operator, keras2onnx.common.OnnxObjectContainer) -> None
    DETECTION_MAX_INSTANCES = 100
    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_MIN_CONFIDENCE = 0.7

    oopb = OnnxOperatorBuilder(container, scope)
    box_transpose = scope.get_unique_variable_name(operator.inputs[0].full_name + '_tx')
    score_transpose = scope.get_unique_variable_name(operator.inputs[1].full_name + '_tx')

    # apply_transpose(scope, operator.inputs[0].full_name, box_transpose, container, perm=[2, 0, 1])
    apply_identity(scope, operator.inputs[0].full_name, box_transpose, container)
    # output shape: [num_batches, spatial_dimension, 4]
    score_identity = scope.get_unique_variable_name(operator.inputs[1].full_name + '_id')
    apply_identity(scope, operator.inputs[1].full_name, score_identity, container)
    # output shape: [num_batches, spatial_dimension, num_classes]

    deltas_transpose = scope.get_unique_variable_name(operator.inputs[2].full_name + '_tx')
    apply_identity(scope, operator.inputs[2].full_name, deltas_transpose, container)
    image_meta = scope.get_unique_variable_name(operator.inputs[3].full_name + '_tx')
    apply_identity(scope, operator.inputs[3].full_name, image_meta, container)
    windows_transpose = norm_boxes_graph(scope, operator, container, oopb, image_meta)
    delta_mul_output = convert_apply_box_deltas_graph(scope, operator, container, oopb, box_transpose, score_identity, deltas_transpose, windows_transpose)

    sliced_score = oopb.add_node('Slice',
                                 [score_identity,
                                  ('_start', oopb.int64, np.array([1], dtype='int64')),
                                  ('_end', oopb.int64, np.array([81], dtype='int64')),
                                  ('_axes', oopb.int64, np.array([2], dtype='int64'))
                                  ],
                                 operator.inputs[1].full_name + '_sliced')
    apply_transpose(scope, sliced_score, score_transpose, container, perm=[0, 2, 1])
    # output shape: [num_batches, num_classes, spatial_dimension]

    max_output_size = scope.get_unique_variable_name('max_output_size')
    iou_threshold = scope.get_unique_variable_name('iou_threshold')
    score_threshold = scope.get_unique_variable_name('layer.score_threshold')

    container.add_initializer(max_output_size, onnx_proto.TensorProto.INT64,
                              [], [DETECTION_MAX_INSTANCES])
    container.add_initializer(iou_threshold, onnx_proto.TensorProto.FLOAT,
                              [], [DETECTION_NMS_THRESHOLD])
    container.add_initializer(score_threshold, onnx_proto.TensorProto.FLOAT,
                              [], [DETECTION_MIN_CONFIDENCE])

    nms_node = next((nd_ for nd_ in operator.nodelist if nd_.type == 'NonMaxSuppressionV3'), operator.nodelist[0])
    nms_output = scope.get_unique_variable_name(operator.output_full_names[0] + '_nms')
    container.add_node("NonMaxSuppression",
                       [delta_mul_output, score_transpose, max_output_size, iou_threshold, score_threshold],
                       nms_output,
                       op_version=operator.target_opset,
                       name=nms_node.name)

    add_init = scope.get_unique_variable_name('add')
    container.add_initializer(add_init, onnx_proto.TensorProto.INT64,
                              [1, 3], [0, 1, 0])
    nms_output_add = scope.get_unique_variable_name(operator.output_full_names[0] + '_class_add')
    container.add_node("Add",
                       [nms_output, add_init],
                       nms_output_add,
                       op_version=operator.target_opset,
                       name=nms_node.name + '_class_idx_add')

    starts_init = scope.get_unique_variable_name('starts')
    ends_init = scope.get_unique_variable_name('ends')
    axes_init = scope.get_unique_variable_name('axes')

    container.add_initializer(starts_init, onnx_proto.TensorProto.INT32,
                              [1], [1])
    container.add_initializer(ends_init, onnx_proto.TensorProto.INT32,
                              [1], [2])
    container.add_initializer(axes_init, onnx_proto.TensorProto.INT32,
                              [1], [1])

    class_idx_output = scope.get_unique_variable_name(operator.output_full_names[0] + '_class_idx')
    container.add_node("Slice",
                       [nms_output_add, starts_init, ends_init, axes_init],
                       class_idx_output,
                       op_version=operator.target_opset,
                       name=nms_node.name+'_class_idx')
    # output shape: [num_selected_indices, 1]

    starts_init_2 = scope.get_unique_variable_name('starts')
    ends_init_2 = scope.get_unique_variable_name('ends')
    axes_init_2 = scope.get_unique_variable_name('axes')

    container.add_initializer(starts_init_2, onnx_proto.TensorProto.INT32,
                              [1], [2])
    container.add_initializer(ends_init_2, onnx_proto.TensorProto.INT32,
                              [1], [3])
    container.add_initializer(axes_init_2, onnx_proto.TensorProto.INT32,
                              [1], [1])

    box_idx_output = scope.get_unique_variable_name(operator.output_full_names[0] + '_box_idx')
    container.add_node("Slice",
                       [nms_output_add, starts_init_2, ends_init_2, axes_init_2],
                       box_idx_output,
                       op_version=operator.target_opset,
                       name=nms_node.name + '_box_idx')
    # output shape: [num_selected_indices, 1]

    box_idx_squeeze = scope.get_unique_variable_name(operator.output_full_names[0] + '_box_idx_squeeze')
    attrs = {'axes': [1]}
    container.add_node("Squeeze",
                       box_idx_output,
                       box_idx_squeeze,
                       op_version=operator.target_opset,
                       name=nms_node.name + '_box_idx_squeeze', **attrs)
    # output shape: [num_selected_indices]

    starts_init_3 = scope.get_unique_variable_name('starts')
    ends_init_3 = scope.get_unique_variable_name('ends')
    axes_init_3 = scope.get_unique_variable_name('axes')
    step_init_3 = scope.get_unique_variable_name('steps')

    container.add_initializer(starts_init_3, onnx_proto.TensorProto.INT32,
                              [1], [2])
    container.add_initializer(ends_init_3, onnx_proto.TensorProto.INT32,
                              [1], [0])
    container.add_initializer(axes_init_3, onnx_proto.TensorProto.INT32,
                              [1], [1])
    container.add_initializer(step_init_3, onnx_proto.TensorProto.INT32,
                              [1], [-1])
    from keras2onnx.common.data_types import Int32TensorType, FloatTensorType
    class_box_idx_output = scope.get_local_variable_or_declare_one(operator.output_full_names[0] + '_class_box_idx',
                                                            type=Int32TensorType(shape=[None, 2]))
    container.add_node("Slice",
                       [nms_output_add, starts_init_3, ends_init_3, axes_init_3, step_init_3],
                       class_box_idx_output.full_name,
                       op_version=operator.target_opset,
                       name=nms_node.name + '_class_box_idx')
    # output shape: [num_selected_indices, 2]

    box_squeeze = scope.get_unique_variable_name(operator.output_full_names[0] + '_box_squeeze')
    attrs = {'axes': [0]}
    container.add_node("Squeeze",
                       delta_mul_output,
                       box_squeeze,
                       op_version=operator.target_opset,
                       name=nms_node.name + '_box_squeeze', **attrs)
    # output shape: [spatial_dimension, 4]

    score_squeeze = scope.get_local_variable_or_declare_one(operator.output_full_names[0] + '_score_squeeze',
                                                             type=FloatTensorType(shape=[None]))
    attrs = {'axes': [0]}
    container.add_node("Squeeze",
                       score_identity,
                       score_squeeze.full_name,
                       op_version=operator.target_opset,
                       name=nms_node.name + '_score_squeeze', **attrs)
    # output shape: [spatial_dimension, num_classes]

    box_gather = scope.get_unique_variable_name(operator.output_full_names[0] + '_box_gather')
    attrs = {'axis': 0}
    container.add_node("Gather",
                       [box_squeeze, box_idx_squeeze],
                       box_gather,
                       op_version=operator.target_opset,
                       name=nms_node.name + '_box_gather', **attrs)
    # output shape: [num_selected_indices, 4]

    score_gather = scope.get_unique_variable_name(operator.output_full_names[0] + '_score_gather')
    container.add_node("GatherND",
                       [score_squeeze.full_name, class_box_idx_output.full_name],
                       score_gather,
                       op_version=operator.target_opset, op_domain='com.microsoft',
                       name=nms_node.name + '_score_gather')
    # output shape: [num_selected_indices]

    score_gather_unsqueeze = scope.get_unique_variable_name(operator.output_full_names[0] + '_score_gather_unsqueeze')
    attrs = {'axes': [1]}
    container.add_node("Unsqueeze",
                       score_gather,
                       score_gather_unsqueeze,
                       op_version=operator.target_opset,
                       name=nms_node.name + '_score_gather_unsqueeze', **attrs)
    # output shape: [num_selected_indices, 1]


    top_k_var = scope.get_unique_variable_name('topK')
    container.add_initializer(top_k_var, onnx_proto.TensorProto.FLOAT,
                              [1], [100.0])

    score_gather_shape = oopb.add_node('Shape',
                                       [score_gather],
                                       operator.inputs[1].full_name + '_score_gather_shape')
    attrs = {'to': 1}
    scope_gather_float = oopb.add_node('Cast',
                                       [score_gather_shape],
                                       operator.inputs[1].full_name + '_scope_gather_float', **attrs)
    top_k_min = oopb.add_node('Min',
                              [scope_gather_float, top_k_var],
                              operator.inputs[1].full_name + '_top_k_min')
    attrs = {'to': 7}
    top_k_min_int = oopb.add_node('Cast',
                                   [top_k_min],
                                   operator.inputs[1].full_name + '_top_k_min_int', **attrs)


    score_top_k_output_val = scope.get_unique_variable_name(operator.output_full_names[0] + '_score_top_k_output_val')
    # output shape: [num_top_K]
    score_top_k_output_idx = scope.get_unique_variable_name(operator.output_full_names[0] + '_score_top_k_output_idx')
    # output shape: [num_top_K]
    attrs = {'axis': 0}
    container.add_node('TopK', [score_gather, top_k_min_int], [score_top_k_output_val, score_top_k_output_idx],
                       op_version=operator.target_opset,
                       name=nms_node.name + '_topK', **attrs)

    class_idx_cast = scope.get_unique_variable_name(operator.output_full_names[0] + '_class_idx_cast')
    attrs = {'to': 1}
    container.add_node('Cast', class_idx_output, class_idx_cast, op_version=operator.target_opset,
                       name=nms_node.name+'_class_idx_cast', **attrs)
    # output shape: [num_selected_indices, 1]

    concat_var = scope.get_unique_variable_name(operator.output_full_names[0] + '_concat_var')
    concat_node = next((nd_ for nd_ in operator.nodelist if nd_.type == 'Concat'), operator.nodelist[0])
    attrs = {'axis': 1}
    container.add_node("Concat",
                       [box_gather, class_idx_cast, score_gather_unsqueeze],
                       concat_var,
                       op_version=operator.target_opset,
                       name=concat_node.name, **attrs)
    # output shape: [num_selected_indices, 6]

    all_gather = scope.get_unique_variable_name(operator.output_full_names[0] + '_all_gather')
    attrs = {'axis': 0}
    container.add_node("Gather",
                       [concat_var, score_top_k_output_idx],
                       all_gather,
                       op_version=operator.target_opset,
                       name=nms_node.name + '_all_gather', **attrs)
    # output shape: [num_top_K, 6]
    padded_result = oopb.add_node('Pad',
                                  [all_gather,
                                   np.array([0, 0, DETECTION_MAX_INSTANCES, 0],
                                            dtype=np.int64)],
                                  nms_node.name + '_padded_result',
                                  op_domain='com.microsoft')
    detection_final = oopb.add_node('Slice',
                                 [padded_result,
                                  ('_start', oopb.int64, np.array([0], dtype='int64')),
                                  ('_end', oopb.int64, np.array([DETECTION_MAX_INSTANCES], dtype='int64')),
                                  ('_axes', oopb.int64, np.array([0], dtype='int64'))
                                  ],
                                 nms_node.name + '_detection_final'
                                 )

    attrs = {'axes': [0]}
    container.add_node("Unsqueeze",
                       detection_final,
                       operator.output_full_names[0],
                       op_version=operator.target_opset,
                       name=nms_node.name + '_concat_unsqueeze', **attrs)
    # output shape: [1, num_top_K, 6]


set_converter(DetectionLayer, convert_DetectionLayer)
set_converter(BatchNorm, convert_BatchNorm)


# Run detection
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def generate_image(images, molded_images, windows, results):
    results_final = []
    for i, image in enumerate(images):
        final_rois, final_class_ids, final_scores, final_masks = \
            model.unmold_detections(results[0][i], results[3][i], # detections[i], mrcnn_mask[i]
                                   image.shape, molded_images[i].shape,
                                   windows[i])
        results_final.append({
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        })
        r = results_final[i]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'])
    return results_final


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Need an image file for object detection.")
        exit(-1)

    model_file_name = './mrcnn.onnx'
    if not os.path.exists(model_file_name):
        # use opset 10 or later
        oml = keras2onnx.convert_keras(model.keras_model, target_opset=10, custom_op_conversions=tf2onnx_contrib_op_conversion)
        onnx.save_model(oml, model_file_name)

    # run with ONNXRuntime
    import onnxruntime
    filename = sys.argv[1]
    image = skimage.io.imread(filename)
    images = [image]

    sess = onnxruntime.InferenceSession(model_file_name)

    # preprocessing
    molded_images, image_metas, windows = model.mold_inputs(images)
    anchors = model.get_anchors(molded_images[0].shape)
    anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

    results = \
        sess.run(None, {"input_image": molded_images.astype(np.float32),
                        "input_anchors": anchors,
                        "input_image_meta": image_metas.astype(np.float32)})

    # postprocessing
    results_final = generate_image(images, molded_images, windows, results)
