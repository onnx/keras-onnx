import os
import sys
import numpy as np
import skimage
import onnx
import keras2onnx
import time

from timeit import default_timer as timer
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

ROOT_DIR = os.path.abspath("./")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


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


from keras2onnx._builtin import on_StridedSlice, on_Round, on_TopKV2, on_Pad, on_CropAndResize, on_GatherNd
from keras2onnx import set_converter
from keras2onnx.ke2onnx.batch_norm import convert_keras_batch_normalization
from keras2onnx.proto import onnx_proto
from keras2onnx.common.onnx_ops import apply_transpose, apply_identity
from mrcnn.model import PyramidROIAlign, BatchNorm, DetectionLayer


def create_onnx_node(scope, operator, container, type):
    # type: (keras2onnx.common.InterimContext, keras2onnx.common.Operator, keras2onnx.common.OnnxObjectContainer, str) -> None
    container.add_node(type, operator.input_full_names, operator.output_full_names, op_version=operator.target_opset)


def convert_PyramidROIAlign(scope, operator, container):
    # create_onnx_node(scope, operator, container, 'PyramidROIAlign')
    #create_onnx_node(scope, operator, container, 'RoiAlign')
    preprocessor_type = 'RoiAlign'
    temp_name_list = []
    for i_ in range(2, 6):
        base_node_name = operator.full_name + '_' + str(i_)
        shape_name = scope.get_unique_variable_name('roi_shape')
        container.add_node('Shape', operator.input_full_names[0], shape_name, op_version=operator.target_opset)

        starts_name = scope.get_unique_variable_name('roi_slice_starts')
        starts = np.asarray([1], dtype=np.int32)
        container.add_initializer(starts_name, onnx_proto.TensorProto.INT32, starts.shape, starts.flatten())

        ends_name = scope.get_unique_variable_name('roi_slice_ends')
        ends = np.asarray([2], dtype=np.int32)
        container.add_initializer(ends_name, onnx_proto.TensorProto.INT32, ends.shape, ends.flatten())

        axes_name = scope.get_unique_variable_name('roi_slice_axes')
        axes = np.asarray([0], dtype=np.int32)
        container.add_initializer(axes_name, onnx_proto.TensorProto.INT32, axes.shape, axes.flatten())

        slice_name = scope.get_unique_variable_name('roi_slice')

        attrs = {'name': base_node_name + '_slice'}
        container.add_node('Slice', [shape_name, starts_name, ends_name, axes_name], slice_name, op_version=operator.target_opset, **attrs)

        constant_of_shape_name = scope.get_unique_variable_name('roi_constant_of_shape')
        attrs = {'name': base_node_name + '_constant_of_shape'}
        container.add_node('ConstantOfShape', slice_name, constant_of_shape_name, op_version=operator.target_opset, **attrs)

        cast_name = scope.get_unique_variable_name('roi_cast')
        attrs = {'name': base_node_name + '_roi_cast', 'to': 7}
        container.add_node('Cast', constant_of_shape_name, cast_name, op_version=operator.target_opset, **attrs)

        squeeze_name = scope.get_unique_variable_name('roi_squeeze')
        attrs = {'name': base_node_name + '_axes', 'axes': [0]}
        container.add_node('Squeeze', operator.input_full_names[0], squeeze_name, op_version=operator.target_opset,
                           **attrs)

        transpose_name = scope.get_unique_variable_name('roi_transpose')
        attrs = {'name': base_node_name + '_transpose', 'perm': [0, 3, 1, 2]}
        container.add_node('Transpose', operator.input_full_names[i_], transpose_name, op_version=operator.target_opset, **attrs)

        temp_name = scope.get_unique_variable_name('pyramid_roi')
        attrs = {'name': scope.get_unique_operator_name(preprocessor_type),
                 'output_height': operator.raw_operator.pool_shape[0],
                 'output_width': operator.raw_operator.pool_shape[1]}
        container.add_node('RoiAlign', [transpose_name, squeeze_name, cast_name], temp_name, op_version=operator.target_opset,
                           **attrs)
        temp_name_list.append(temp_name)

    attrs = {'name': scope.get_unique_operator_name(preprocessor_type) + '_concat', 'axis': 0}
    container.add_node('Concat', temp_name_list, operator.output_full_names, op_version=operator.target_opset, **attrs)


def convert_BatchNorm(scope, operator, container):
    convert_keras_batch_normalization(scope, operator, container)


from keras2onnx.common.onnx_ops import OnnxOperatorBuilder

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
    '''
    concat_result = scope.get_unique_variable_name(operator.output_full_names[0] + '_concat_result')
    attrs = {'axis': 1}
    container.add_node("Concat",
                       [y1, x1, y2, x2],
                       concat_result,
                       op_version=operator.target_opset,
                       name=operator.outputs[0].full_name + '_concat_result', **attrs)
    '''
    # output shape: [spatial_dimension, 4]
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
    '''
    indices_float = scope.get_unique_variable_name('class_ids_float')
    attrs = {'to': 1}
    container.add_node('Cast', indices, indices_float, op_version=operator.target_opset,
                       **attrs)
    attrs = {'axis': 1}
    indices_acc = oopb.add_node('Concat',
                         [concat_result,
                          indices_float
                          ],
                         operator.inputs[1].full_name + '_indices_acc', **attrs)
    # output shape: [spatial_dimension, 6]
    indices_cut = oopb.add_node('Slice',
                         [indices_acc,
                          ('_start', oopb.int64, np.array([0], dtype='int64')),
                          ('_end', oopb.int64, np.array([1000], dtype='int64')),
                          ('_axes', oopb.int64, np.array([0], dtype='int64'))
                          ],
                         operator.inputs[1].full_name + '_slcie_slice')
    concat_unsqueeze = scope.get_unique_variable_name('concat_unsqueeze')
    attrs = {'axes': [0]}
    container.add_node('Unsqueeze', indices_cut, concat_unsqueeze, op_version=operator.target_opset,
                       **attrs)
    return concat_unsqueeze
    '''
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

    oopb = OnnxOperatorBuilder(container, scope)
    # type: (keras2onnx.common.InterimContext, keras2onnx.common.Operator, keras2onnx.common.OnnxObjectContainer) -> None
    DETECTION_MAX_INSTANCES = 100
    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_MIN_CONFIDENCE = 0.7

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


    # box_batch = scope.get_unique_variable_name(operator.inputs[0].full_name + '_btc')
    # score_batch = scope.get_unique_variable_name(operator.inputs[1].full_name + '_btc')

    #container.add_node("Unsqueeze", box_transpose,
    #                   box_batch, op_version=operator.target_opset, axes=[0])
    # container.add_node("Unsqueeze", score_transpose,
    #                   score_batch, op_version=operator.target_opset, axes=[0])

    max_output_size = scope.get_unique_variable_name('max_output_size')
    iou_threshold = scope.get_unique_variable_name('iou_threshold')
    score_threshold = scope.get_unique_variable_name('layer.score_threshold')

    container.add_initializer(max_output_size, onnx_proto.TensorProto.INT64,
                              [], [DETECTION_MAX_INSTANCES])
    container.add_initializer(iou_threshold, onnx_proto.TensorProto.FLOAT,
                              [], [DETECTION_NMS_THRESHOLD])
    container.add_initializer(score_threshold, onnx_proto.TensorProto.FLOAT,
                              [], [DETECTION_MIN_CONFIDENCE])


    nms_node = next((nd_ for nd_ in operator.node_list if nd_.type == 'NonMaxSuppressionV3'), operator.node_list[0])
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
    #class_box_idx_output = scope.get_unique_variable_name(operator.output_full_names[0] + '_class_box_idx')
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

    #score_squeeze = scope.get_unique_variable_name(operator.output_full_names[0] + '_score_squeeze')
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
    '''
    container.add_initializer(top_k_var, onnx_proto.TensorProto.INT64,
                              [1], [100])
    '''
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
    # container.add_node('TopK', [score_gather, top_k_var], [score_top_k_output_val, score_top_k_output_idx], op_version=operator.target_opset,
    container.add_node('TopK', [score_gather, top_k_min_int], [score_top_k_output_val, score_top_k_output_idx],
                       op_version=operator.target_opset,
                       name=nms_node.name + '_topK', **attrs)

    # nonzero_classid = oopb.add_node('NonZero',
    #                                 [class_idx_output],
    #                                 nms_node.name + '_nonzero_classid',
    #                                 )
    # nonzero_reshaped = oopb.add_node('Reshape',
    #                                  [nonzero_classid,
    #                                   ('shape', oopb.int64, np.array([-1]))],
    #                                  nms_node.name + '_nonzero_reshaped')
    #
    # intersection_idx = oopb.add_node('DenseToDenseSetOperation',
    #                                            [nonzero_reshaped, score_top_k_output_idx],
    #                                            nms_node.name + '_intersection',
    #                                            op_domain='com.microsoft')

    '''
    box_unsqueeze = scope.get_unique_variable_name(operator.output_full_names[0] + '_box_unsqueeze')
    attrs = {'axes': [1]}
    container.add_node("Unsqueeze",
                       box_gather,
                       box_unsqueeze,
                       op_version=operator.target_opset,
                       name=nms_node.name + '_box_unsqueeze', **attrs)

    score_unsqueeze = scope.get_unique_variable_name(operator.output_full_names[0] + '_score_unsqueeze')
    attrs = {'axes': [1]}
    container.add_node("Unsqueeze",
                       score_gather,
                       score_unsqueeze,
                       op_version=operator.target_opset,
                       name=nms_node.name + '_score_unsqueeze', **attrs)
    '''
    class_idx_cast = scope.get_unique_variable_name(operator.output_full_names[0] + '_class_idx_cast')
    attrs = {'to': 1}
    container.add_node('Cast', class_idx_output, class_idx_cast, op_version=operator.target_opset,
                       name=nms_node.name+'_class_idx_cast', **attrs)
    # output shape: [num_selected_indices, 1]

    '''
    cast_name = scope.get_unique_variable_name(operator.output_full_names[0] + '_nms_cast')
    attrs = {'to': 1}
    container.add_node('Cast', nms_output, cast_name, op_version=operator.target_opset, **attrs)
    '''
    concat_var = scope.get_unique_variable_name(operator.output_full_names[0] + '_concat_var')
    concat_node = next((nd_ for nd_ in operator.node_list if nd_.type == 'Concat'), operator.node_list[0])
    attrs = {'axis': 1}
    container.add_node("Concat",
                       [box_gather, class_idx_cast, score_gather_unsqueeze],
                       concat_var,
                       #operator.output_full_names[0],
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
    #padded_result = oopb.add_node('DynamicPad',
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

    '''
    starts_init_4 = scope.get_unique_variable_name('starts')
    ends_init_4 = scope.get_unique_variable_name('ends')
    axes_init_4 = scope.get_unique_variable_name('axes')

    container.add_initializer(starts_init_4, onnx_proto.TensorProto.INT32,
                              [1], [4])
    container.add_initializer(ends_init_4, onnx_proto.TensorProto.INT32,
                              [1], [5])
    container.add_initializer(axes_init_4, onnx_proto.TensorProto.INT32,
                              [1], [1])

    concat_slice = scope.get_unique_variable_name(operator.output_full_names[0] + '_concat_slice')
    container.add_node("Slice",
                       [all_gather, starts_init_4, ends_init_4, axes_init_4],
                       concat_slice,
                       op_version=operator.target_opset,
                       name=nms_node.name+'_concat_slice')
    # output shape: [num_selected_indices, 1]

    concat_non_zero = scope.get_unique_variable_name(operator.output_full_names[0] + '_concat_non_zero')
    container.add_node("NonZero",
                       concat_slice,
                       concat_non_zero,
                       op_version=operator.target_opset,
                       name=nms_node.name+'_concat_non_zero')
    # output shape: [2, num_nonzero_indices]

    starts_init_5 = scope.get_unique_variable_name('starts')
    ends_init_5 = scope.get_unique_variable_name('ends')
    axes_init_5 = scope.get_unique_variable_name('axes')

    container.add_initializer(starts_init_5, onnx_proto.TensorProto.INT32,
                              [1], [0])
    container.add_initializer(ends_init_5, onnx_proto.TensorProto.INT32,
                              [1], [1])
    container.add_initializer(axes_init_5, onnx_proto.TensorProto.INT32,
                              [1], [1])

    concat_non_zero_slice = scope.get_unique_variable_name(operator.output_full_names[0] + '_concat_non_zero_slice')
    container.add_node("Slice",
                       [concat_non_zero, starts_init_5, ends_init_5, axes_init_5],
                       concat_non_zero_slice,
                       op_version=operator.target_opset,
                       name=nms_node.name+'_concat_non_zero_slice')
    # output shape: [1, num_nonzero_indices]

    concat_non_zero_squeeze = scope.get_unique_variable_name(operator.output_full_names[0] + '_concat_non_zero_squeeze')
    attrs = {'axes': [0]}
    container.add_node("Squeeze",
                       concat_non_zero_slice,
                       concat_non_zero_squeeze,
                       op_version=operator.target_opset,
                       name=nms_node.name + '_concat_non_zero_squeeze', **attrs)
    # output shape: [num_nonzero_indices]

    cocat_gather = scope.get_unique_variable_name(operator.output_full_names[0] + '_cocat_gather')
    attrs = {'axis': 0}
    container.add_node("Gather",
                       [concat_var, concat_non_zero_squeeze],
                       cocat_gather,
                       op_version=operator.target_opset,
                       name=nms_node.name + '_cocat_gather', **attrs)
    # output shape: [num_nonzero_indices, 6]

    pad_concat = scope.get_unique_variable_name(operator.output_full_names[0] + '_pad_concat')
    pad_param = scope.get_unique_variable_name('pad')
    container.add_initializer(pad_param, onnx_proto.TensorProto.INT64,
                              [1, 4], [0, 0, 99, 0])
    # container.add_node("DynamicPad",
    container.add_node("Pad",
                       [cocat_gather, pad_param],
                       pad_concat,
                       op_version=operator.target_opset, op_domain='com.microsoft',
                       name=nms_node.name+'_pad_concat')
    '''

    attrs = {'axes': [0]}
    container.add_node("Unsqueeze",
                       # pad_concat,
                       detection_final,
                       operator.output_full_names[0],
                       op_version=operator.target_opset,
                       name=nms_node.name + '_concat_unsqueeze', **attrs)
    # output shape: [1, num_top_K, 6]

# set_converter(PyramidROIAlign, convert_PyramidROIAlign)
set_converter(DetectionLayer, convert_DetectionLayer)
set_converter(BatchNorm, convert_BatchNorm)

_custom_op_handlers = {
    'Round': (on_Round, []),
    'StridedSlice': (on_StridedSlice, []),
    'TopKV2': (on_TopKV2, []),
    'Pad': (on_Pad, []),
    'PadV2': (on_Pad, []),
    'CropAndResize': (on_CropAndResize, []),
    'GatherNd': (on_GatherNd, [])
}

# Run detection
#results = model.detect([image], verbose=1)
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

'''
for res in results:
    print("rois: " + str(res['rois']))
    print(", ".join('class :' + class_names[id_] for id_ in res['class_ids']))
    print("scores: " + str(res['scores']))
    # print("masks: " + str(res['masks']))
'''

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

if len(sys.argv) > 1 and sys.argv[1] == '-c':
    model.keras_model.save('mrcnn.h5')
    oml = keras2onnx.convert_keras(model.keras_model, target_opset=10, debug_mode=True, custom_op_conversions=_custom_op_handlers)
    onnx.save_model(oml, './mrcnn.onnx')
else:
    # run with ONNXRuntime
    import onnxruntime
    print("The python process id is %d, attach the VS as the debugger." % os.getpid())

    from os import walk

    f_list = []
    mypath = 'E:/test2017/test2017/'
    for (dirpath, dirnames, filenames) in walk(mypath):
        f_list.extend(filenames)
        break

    actual_count = 0
    total_count = 0
    correct_count = 0
    correct_count_box = 0
    correct_count_mask = 0
    total_keras_time = 0
    total_onnx_time = 0
    skip_count = 0
    process_generate_image = True
    enable_skip = True

    # process_generate_image = True

    # sess = onnxruntime.InferenceSession('./mrcnn_edit.onnx')
    sess = onnxruntime.InferenceSession('./mrcnn_resnet_50.onnx')

    for filename in f_list:
        # Load a random image from the images folder
        # image = skimage.io.imread('../data/elephant.jpg')
        if enable_skip:
            skip_count = skip_count + 1
            if skip_count < 26:
                continue
        actual_count = actual_count + 1
        print(filename)

        loading_pass = True
        try:
            #image = skimage.io.imread(mypath + filename)
            image = skimage.io.imread('../data/elephant.jpg')
            images = [image]
        except Exception as ex:
            print("loading image fails: " + filename)
            loading_pass = False

        if not loading_pass:
            continue

        keras_pass = True
        try:
            molded_images, image_metas, windows = model.mold_inputs(images)
            anchors = model.get_anchors(molded_images[0].shape)
            anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)
            start = timer()
            results_k = model.keras_model.predict(
                [molded_images.astype(np.float32), image_metas.astype(np.float32), anchors])
            keras_end_time = timer()
            total_keras_time = total_keras_time + keras_end_time - start
            print("Total Keras inf time is %fs " % (keras_end_time - start))
        except Exception as ex:
            print("the image fails on keras: " + filename)
            keras_pass = False

        if not keras_pass:
            continue


        print("keras runs good: " + filename)
        # sess = onnxruntime.InferenceSession('./mrcnn.onnx')
        onnx_pass = True
        '''
        results = \
            sess.run(None, {"input_image:01": molded_images.astype(np.float32),
                            "input_anchors:01": anchors,
                            "input_image_meta:01": image_metas.astype(np.float32)})
        '''
        try:
            start_ort = timer()
            '''
            from onnx import numpy_helper
            tensor1 = numpy_helper.from_array(molded_images.astype(np.float32))
            tensor1.name = 'input_image:01'
            with open(os.path.join('test_data_set_0', 'input_0.pb'), 'wb') as f:
                f.write(tensor1.SerializeToString())
            tensor2 = numpy_helper.from_array(anchors)
            tensor2.name = 'input_anchors:01'
            with open(os.path.join('test_data_set_0', 'input_1.pb'), 'wb') as f:
                f.write(tensor2.SerializeToString())
            tensor3 = numpy_helper.from_array(image_metas.astype(np.float32))
            tensor3.name = 'input_image_meta:01'
            with open(os.path.join('test_data_set_0', 'input_2.pb'), 'wb') as f:
                f.write(tensor3.SerializeToString())
            '''
            results =\
                sess.run(None, {"input_image:01": molded_images.astype(np.float32),
                                "input_anchors:01": anchors,
                                "input_image_meta:01": image_metas.astype(np.float32)})
            '''
            tensor_output_1 = numpy_helper.from_array(results[0], 'mrcnn_detection/Reshape_1:0')
            with open(os.path.join('test_data_set_0', 'output_0.pb'), 'wb') as f:
                f.write(tensor_output_1.SerializeToString())
            tensor_output_2 = numpy_helper.from_array(results[1], 'mrcnn_class/Reshape_1:0')
            with open(os.path.join('test_data_set_0', 'output_1.pb'), 'wb') as f:
                f.write(tensor_output_2.SerializeToString())
            tensor_output_3 = numpy_helper.from_array(results[2], 'mrcnn_bbox/Reshape:0')
            with open(os.path.join('test_data_set_0', 'output_2.pb'), 'wb') as f:
                f.write(tensor_output_3.SerializeToString())
            tensor_output_4 = numpy_helper.from_array(results[3], 'mrcnn_mask/Reshape_1:0')
            with open(os.path.join('test_data_set_0', 'output_3.pb'), 'wb') as f:
                f.write(tensor_output_4.SerializeToString())
            tensor_output_5 = numpy_helper.from_array(results[4], 'ROI/packed_2:0')
            with open(os.path.join('test_data_set_0', 'output_4.pb'), 'wb') as f:
                f.write(tensor_output_5.SerializeToString())
            tensor_output_6 = numpy_helper.from_array(results[5], 'rpn_class/concat:0')
            with open(os.path.join('test_data_set_0', 'output_5.pb'), 'wb') as f:
                f.write(tensor_output_6.SerializeToString())
            tensor_output_7 = numpy_helper.from_array(results[6], 'rpn_bbox/concat:0')
            with open(os.path.join('test_data_set_0', 'output_6.pb'), 'wb') as f:
                f.write(tensor_output_7.SerializeToString())
            '''
            onnx_end_time = timer()
            total_onnx_time = total_onnx_time + onnx_end_time - start
            print("Total ORT inf time is %fs " % (onnx_end_time - start))
        except Exception as ex:
            print("the image fails on onnx: " + filename)
            onnx_pass = False

        if not onnx_pass:
            continue

        print("onnxruntime runs good: " + filename)

        rtol = 1.e-2 #1.e-3
        atol = 1.e-3 #1.e-6
        # compare_idx = range(len(results))
        compare_idx = [0, 3]
        result_match = True
        box_match = False
        mask_match = False

        for n_ in compare_idx:
            expected_list = results_k[n_].flatten()
            actual_list = results[n_].flatten()
            if n_ == 0:
                expected_class_id = expected_list[4::6]
                actual_class_id = actual_list[4::6]
                expected_idx_sorted = sorted(range(len(expected_class_id)), key=lambda k: expected_class_id[k], reverse=True)
                actual_idx_sorted = sorted(range(len(actual_class_id)), key=lambda k: actual_class_id[k], reverse=True)
                expected_list_copy = np.copy(expected_list)
                actual_list_copy = np.copy(actual_list)
                for i_ in range(len(expected_idx_sorted)):
                    expected_list_copy[6*i_:6*(i_+1)] = expected_list[6*expected_idx_sorted[i_]:6*(expected_idx_sorted[i_]+1)]
                for i_ in range(len(actual_idx_sorted)):
                    actual_list_copy[6*i_:6*(i_+1)] = actual_list[6*actual_idx_sorted[i_]:6*(actual_idx_sorted[i_]+1)]
                expected_list = expected_list_copy
                actual_list = actual_list_copy

            if n_ == 3:
                '''
                final_rois_onnx, final_class_ids_onnx, final_scores_onnx, final_masks_onnx = \
                    model.unmold_detections(results[0][0], results[3][0],  # detections[i], mrcnn_mask[i]
                                            image.shape, molded_images[0].shape,
                                            windows[0])
                final_masks_onnx = final_masks_onnx.astype(int)
                total_mask_onnx = np.zeros((final_masks_onnx.shape[0], final_masks_onnx.shape[1]), dtype=int)
                for mask_idx_ in range(final_masks_onnx.shape[2]):
                    total_mask_onnx = total_mask_onnx + final_masks_onnx[:, :, mask_idx_]
                final_rois_keras, final_class_ids_keras, final_scores_keras, final_masks_keras = \
                    model.unmold_detections(results_k[0][0], results_k[3][0],  # detections[i], mrcnn_mask[i]
                                            image.shape, molded_images[0].shape,
                                            windows[0])
                final_masks_keras = final_masks_keras.astype(int)
                total_mask_keras = np.zeros((final_masks_keras.shape[0], final_masks_keras.shape[1]), dtype=int)
                for mask_idx_ in range(final_masks_keras.shape[2]):
                    total_mask_keras = total_mask_keras + final_masks_keras[:, :, mask_idx_]

                area_intersection = np.logical_and(total_mask_keras, total_mask_onnx)
                area_union = np.logical_or(total_mask_keras, total_mask_onnx)
                count_intersection = np.count_nonzero(area_intersection)
                count_union = np.count_nonzero(area_union)
                '''
                expected_list = np.array([1 if expected > 0.999 else 0 for expected in expected_list])
                actual_list = np.array([1 if actual > 0.999 else 0 for actual in actual_list])

            diff_list = abs(expected_list - actual_list)
            count_total = len(expected_list)
            count_error = 0
            cur_count = 0

            res = np.allclose(actual_list, expected_list, rtol=rtol, atol=atol)
            if not res:
                result_match = False

            if n_ == 0 and res:
                box_match = True
                correct_count_box = correct_count_box + 1
            if n_ == 3:
                union_list = expected_list + actual_list
                intersection_list = np.array([1 if union > 1.999 else 0 for union in union_list])
                count_keras = np.count_nonzero(expected_list)
                count_onnx = np.count_nonzero(actual_list)                
                count_intersection = np.count_nonzero(intersection_list)
                count_union = np.count_nonzero(union_list)
                ratio_str = str(count_intersection / count_union) if count_union > 0 else '1'
                print("case = "  + ", keras count = " +
                      str(count_keras) + ", onnx count = " +
                      str(count_onnx) + ", intersection count = " +
                      str(count_intersection) + ", union count = " + str(count_union) + ", with ratio = " +  ratio_str,
                      file=sys.stderr)
                if count_intersection > 0.99 * count_union or count_union == 0:
                    mask_match = True
                    correct_count_mask = correct_count_mask + 1

            count_error_threshold = 20
            if n_ != 3:
                for e_, a_, d_ in zip(expected_list, actual_list, diff_list):
                    if d_ > atol + rtol * abs(a_):
                        if count_error < count_error_threshold:  # print the first count_error_threshold mismatches
                            print(
                                "case = " + ", result mismatch for results_keras = " + str(e_) +
                                ", results_onnx = " + str(a_) + ", cur_count = " + str(cur_count), file=sys.stderr)
                        count_error = count_error + 1
                    cur_count = cur_count + 1
                print("case = "  + ", " +
                      str(count_error) + " mismatches out of " + str(count_total) + " for list " + str(n_),
                      file=sys.stderr)

            if n_ == 3:
                for e_, a_, d_ in zip(expected_list, actual_list, diff_list):
                    if d_ > 0:
                        if count_error < count_error_threshold:  # print the first count_error_threshold mismatches
                            print(
                                "case = " + ", result mismatch for results_keras = " + str(e_) +
                                ", results_onnx = " + str(a_) + ", cur_count = " + str(cur_count), file=sys.stderr)
                        count_error = count_error + 1
                    cur_count = cur_count + 1
                print("case = "  + ", " +
                      str(count_error) + " mismatches out of " + str(count_total) + " for list " + str(n_),
                      file=sys.stderr)

        try:
            results_final_k = generate_image(images, molded_images, windows, results_k)
            results_final = generate_image(images, molded_images, windows, results)
        except Exception as ex:
            print("the image cannot be generated for file:" + filename)

        '''
        if result_match:
            correct_count = correct_count + 1
        elif process_generate_image:
            try:
                results_final_k = generate_image(images, molded_images, windows, results_k)
                results_final = generate_image(images, molded_images, windows, results)
            except Exception as ex:
                print("the image cannot be generated for file:" + filename)
        '''
        total_count = total_count + 1
        print("actual_count = " + str(actual_count) + ", total_count = " + str(total_count) + ", correct_count = " + str(correct_count)
              + ", correct_count_box = " + str(correct_count_box) + ", correct_count_mask = " + str(correct_count_mask) )

        if total_count > 0:
            print("avg_keras_time = " + str(total_keras_time / total_count) + ", avg_onnx_time = " + str(total_onnx_time / total_count)
                  + ", ratio = " + str(total_onnx_time / total_keras_time))

        #time.sleep(5)
        #break
        if actual_count == 100:
            break

