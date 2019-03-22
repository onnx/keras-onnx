import colorsys
import os
import sys
import inspect
from timeit import default_timer as timer

import onnx
import numpy as np
import tensorflow as tf
import keras
from PIL import Image, ImageFont, ImageDraw
from keras import backend as K
from keras.layers import Input
from keras.models import load_model
from keras2onnx import convert_keras
from keras2onnx import set_converter
from keras2onnx.common.onnx_ops import apply_transpose
from keras2onnx.proto import onnx_proto
from tf2onnx import utils

import yolo3
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body, yolo_boxes_and_scores
from yolo3.utils import letterbox_image


class YOLOEvaluationLayer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(YOLOEvaluationLayer, self).__init__()
        self.max_boxes = kwargs.get('max_boxes', 20)
        self.score_threshold = kwargs.get('score_threshold', .6)
        self.iou_threshold = kwargs.get('iou_threshold', .5)
        self.anchors = np.array(kwargs.get('anchors'))
        self.num_classes = kwargs.get('num_classes')

    def get_config(self):
        config = {
            "max_boxes": self.max_boxes,
            "score_threshold": self.score_threshold,
            "iou_threshold": self.iou_threshold,
            "anchors": self.anchors,
            "num_classes": self.num_classes,
        }

        return config

    def call(self, inputs, **kwargs):
        """Evaluate YOLO model on given input and return filtered boxes."""
        yolo_outputs, input_image_shape = (inputs[0:3], inputs[3])
        num_layers = len(yolo_outputs)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5],
                                                                                 [1, 2, 3]]  # default setting
        input_shape = K.shape(yolo_outputs[0])[1:3] * 32
        boxes = []
        box_scores = []
        for l in range(num_layers):
            _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], self.anchors[anchor_mask[l]], self.num_classes,
                                                        input_shape, input_image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = K.concatenate(boxes, axis=0)
        box_scores = K.concatenate(box_scores, axis=0)
        return [boxes, box_scores]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return [(None, 4, None), (None, None)]


class YOLONMSLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(YOLONMSLayer, self).__init__()
        self.max_boxes = kwargs.get('max_boxes', 20)
        self.score_threshold = kwargs.get('score_threshold', .6)
        self.iou_threshold = kwargs.get('iou_threshold', .5)
        self.num_classes = kwargs.get('num_classes')

    def get_config(self):
        config = {
            "max_boxes": self.max_boxes,
            "score_threshold": self.score_threshold,
            "iou_threshold": self.iou_threshold,
            "num_classes": self.num_classes,
        }

        return config

    def call(self, inputs, **kwargs):
        boxes = inputs[0]
        box_scores = inputs[1]

        mask = box_scores >= self.score_threshold
        max_boxes_tensor = K.constant(self.max_boxes, dtype='int32')
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(self.num_classes):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=self.iou_threshold)
            class_boxes = K.gather(class_boxes, nms_index)
            class_box_scores = K.gather(class_box_scores, nms_index)
            classes = K.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = K.concatenate(boxes_, axis=0)
        scores_ = K.concatenate(scores_, axis=0)
        classes_ = K.concatenate(classes_, axis=0)

        boxes_r = tf.expand_dims(tf.expand_dims(boxes_, 0), 0)
        scores_r = tf.expand_dims(tf.expand_dims(scores_, 0), 0)
        return [boxes_r, scores_r]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return [(None, None, None, 4), (None, None, None)]


class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/yolo.h5'  # model path or trained weights path
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()
        self.session = None
        self.session_final = None
        self.final_model = None
        K.set_learning_phase(0)

    @staticmethod
    def _get_data_path(name):
        path = os.path.expanduser(name)
        if not os.path.isabs(path):
            yolo3_dir = os.path.dirname(inspect.getabsfile(yolo3))
            path = os.path.join(yolo3_dir, os.path.pardir, path)
        return path

    def _get_class(self):
        classes_path = self._get_data_path(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = self._get_data_path(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_model(self):
        model_path = self._get_data_path(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        input_image_shape = keras.Input(shape=(2,))
        y1 = keras.Input((None, None, 255), dtype='float32')
        y2 = keras.Input((None, None, 255), dtype='float32')
        y3 = keras.Input((None, None, 255), dtype='float32')

        boxes, box_scores = \
            YOLOEvaluationLayer(anchors=self.anchors, num_classes=len(self.class_names))(
                inputs=[y1, y2, y3, input_image_shape])

        out_boxes, out_scores = \
            YOLONMSLayer(anchors=self.anchors, num_classes=len(self.class_names))(
                inputs=[boxes, box_scores])

        self.final_model = keras.Model(inputs=[y1, y2, y3, input_image_shape],
                                       outputs=[out_boxes, out_scores])
        self.final_model.save('model_data/final_model.h5')
        print('{} model, anchors, and classes loaded.'.format(model_path))

    def generate(self):
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting

        last_dim = num_anchors / 3 * (num_classes + 5)
        self.i0 = K.placeholder(shape=(None, None, None, last_dim))
        self.i1 = K.placeholder(shape=(None, None, None, last_dim))
        self.i2 = K.placeholder(shape=(None, None, None, last_dim))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval([self.i0, self.i1, self.i2], self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)

        return boxes, scores, classes

    def detect_with_onnx(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data = np.transpose(image_data, [2, 0, 1])

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        r = self.session.run({'input_1_0': image_data},
                                ['conv2d_59_BiasAdd_01',
                                 'conv2d_67_BiasAdd_01',
                                 'conv2d_75_BiasAdd_01'])

        feed_f = dict(zip((n_.name for n_ in self.session_final.get_inputs()),
             (r[0],
              r[1],
              r[2],
              [image.size[1], image.size[0]])))
        all_boxes, all_scores = self.session_final.run(feed_f)

        out_boxes, out_scores, out_classes = [], [], []
        for idx_, sc_ in np.ndenumerate(all_scores):
            if sc_ > 0.0:
                out_classes.append(idx_[1])
                out_scores.append(all_scores[idx_])
                out_boxes.append(all_boxes[idx_])

        """
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.i0: r[0],
                self.i1: r[1],
                self.i2: r[2],
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        """

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print("time=", end - start)
        return image


def detect_img(yolo, name):
    import onnxruntime
    image = Image.open(name)
    yolo.session = onnxruntime.InferenceSession('model_data/yolov3_0.onnx')
    yolo.session_final = onnxruntime.InferenceSession('model_data/yolov3_1.onnx')
    r_image = yolo.detect_with_onnx(image)

    n_ext = name.rindex('.')
    score_file = name[0:n_ext] + '_score' + name[n_ext:]
    r_image.save(score_file, "JPEG")

def on_StridedSlice(ctx, node, name, args):
    # node.type = "Reverse"
    # for now we implement common cases. Things like strides!=1 are not mappable to onnx.
    not_supported_attr = ["new_axis_mask"]
    for attr_name in not_supported_attr:
        attr = node.get_attr(attr_name)
        if attr is not None and attr.i != 0:
            raise ValueError("StridedSlice: attribute " + attr_name + " not supported")
    input_shape = ctx.get_shape(node.input[0])
    begin = node.inputs[1].get_tensor_value()
    end = node.inputs[2].get_tensor_value()
    strides = node.inputs[3].get_tensor_value()
    max_size = sys.maxsize
    begin_mask = node.get_attr("begin_mask")
    begin_mask = begin_mask.i if begin_mask is not None else 0
    end_mask = node.get_attr("end_mask")
    end_mask = end_mask.i if end_mask is not None else 0
    ellipsis_mask = node.get_attr("ellipsis_mask")
    ellipsis_mask = ellipsis_mask.i if ellipsis_mask is not None else 0
    shrink_axis_mask = node.get_attr("shrink_axis_mask")
    shrink_axis_mask = shrink_axis_mask.i if shrink_axis_mask is not None else 0
    new_begin = []
    new_end = []
    axes = []
    # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
    needs_squeeze = []
    reverse_axes = []
    for idx, begin_item in enumerate(begin):
        end_item = end[idx]
        if strides[idx] == -1:
            reverse_axes.append(idx)
        # if strides[idx] != 1:
        # raise ValueError("StridedSlice: only strides=1 is supported, current stride =" + str(strides[idx]))
        axes.append(idx)

        if (begin_mask >> idx) & 1 != 0 and (end_mask >> idx) & 1 != 0:
            new_begin.append(0)
            new_end.append(max_size)
            continue

        if begin_item == 0 and end_item == 0:
            new_begin.append(0)
            new_end.append(max_size)
            continue

        # an implicit condition is stride == 1 (checked in above)
        if begin_item < 0 and end_item == 0:
            end_item = max_size

        mask = (shrink_axis_mask >> idx) & 1
        if mask != 0:
            new_begin.append(begin_item)
            new_end.append(end_item)
            if begin_item == 0 and end_item == 0:
                aa = 1
            needs_squeeze.append(idx)
            continue

        if (begin_mask >> idx) & 1 != 0:
            new_begin.append(0)
            new_end.append(end_item)
            if end_item == 0:
                aa = 1
            continue

        if (end_mask >> idx) & 1 != 0:
            new_begin.append(begin_item)
            new_end.append(max_size)
            continue

        if begin_item == 0 and end_item == 0:
            aa = 1
        new_begin.append(begin_item)
        new_end.append(end_item)

    node.set_attr("starts", new_begin)
    node.set_attr("ends", new_end)
    node.set_attr("axes", axes)
    node.type = "Slice"
    ctx.remove_input(node, node.input[3])
    ctx.remove_input(node, node.input[2])
    ctx.remove_input(node, node.input[1])
    nodes = [node]
    use_reverse_op = False
    reverse_flag = False
    if use_reverse_op and len(reverse_axes) > 0:
        name = utils.make_name(node.name)
        name = name + '_reverse'
        reverse_node = ctx.insert_new_node_on_output("Reverse", node.output[0], name)
        reverse_node.set_attr("axes", reverse_axes)
        reverse_node.domain = 'com.microsoft'
        nodes.append(reverse_node)
        input_dtype = ctx.get_dtype(node.output[0])
        ctx.set_dtype(reverse_node.output[0], input_dtype)
        ctx.copy_shape(node.output[0], reverse_node.output[0])
        reverse_flag = True

    if needs_squeeze:
        name = utils.make_name(node.name)
        if use_reverse_op:
            if reverse_flag:
                squeeze_node = ctx.insert_new_node_on_output("Squeeze", reverse_node.output[0], name)
            else:
                squeeze_node = ctx.insert_new_node_on_output("Squeeze", node.output[0], name)
        else:
            squeeze_node = ctx.insert_new_node_on_output("Squeeze", node.output[0], name)
        squeeze_node.set_attr("axes", needs_squeeze)
        nodes.append(squeeze_node)
        input_dtype = ctx.get_dtype(node.output[0])
        ctx.set_dtype(squeeze_node.output[0], input_dtype)
        ctx.copy_shape(node.output[0], squeeze_node.output[0])

    # onnx slice as of opset 7 does only take float tensors ... cast if needed
    '''
    input_dtype = ctx.get_dtype(node.input[0])
    if input_dtype != onnx_pb.TensorProto.FLOAT:
        if node.inputs[0].type == "Cast":
            # override the previous cast
            cast_node = node.inputs[0]
        else:
            cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[0])
            nodes.insert(0, cast_node)
        cast_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
        ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
        ctx.copy_shape(node.input[0], cast_node.output[0])
        # undo the cast afer slice
        name = utils.make_name(node.name)
        cast_node = ctx.insert_new_node_on_output("Cast", nodes[-1].output[0], name)
        cast_node.set_attr("to", input_dtype)
        ctx.set_dtype(cast_node.output[0], input_dtype)
        ctx.copy_shape(node.output[0], cast_node.output[0])
        nodes.append(cast_node)
    '''
    return nodes


def on_Round(ctx, node, name, args):
    node.type = "Ceil"
    return node


_custom_op_handlers = {
    'Round': (on_Round, []),
    'StridedSlice': (on_StridedSlice, [])}


def create_onnx_node(scope, operator, container, type):
    # type: (keras2onnx.common.InterimContext, keras2onnx.common.Operator, keras2onnx.common.OnnxObjectContainer, str) -> None

    container.add_node(type, operator.input_full_names, operator.output_full_names, op_version=operator.target_opset)


def convert_NMSLayer(scope, operator, container):
    # type: (keras2onnx.common.InterimContext, keras2onnx.common.Operator, keras2onnx.common.OnnxObjectContainer) -> None
    box_transpose = scope.get_unique_variable_name(operator.inputs[0].full_name + '_tx')
    score_transpose = scope.get_unique_variable_name(operator.inputs[1].full_name + '_tx')

    apply_transpose(scope, operator.inputs[0].full_name, box_transpose, container, perm=[2, 0, 1])
    apply_transpose(scope, operator.inputs[1].full_name, score_transpose, container, perm=[1, 0])

    box_batch = scope.get_unique_variable_name(operator.inputs[0].full_name + '_btc')
    score_batch = scope.get_unique_variable_name(operator.inputs[1].full_name + '_btc')

    container.add_node("Unsqueeze", box_transpose,
                       box_batch, op_version=operator.target_opset, axes=[0])
    container.add_node("Unsqueeze", score_transpose,
                       score_batch, op_version=operator.target_opset, axes=[0])

    layer = operator.raw_operator  # type: YOLONMSLayer

    max_output_size = scope.get_unique_variable_name('max_output_size')
    iou_threshold = scope.get_unique_variable_name('iou_threshold')
    score_threshold = scope.get_unique_variable_name('layer.score_threshold')

    container.add_initializer(max_output_size, onnx_proto.TensorProto.INT32,
                              [1], [layer.max_boxes])
    container.add_initializer(iou_threshold, onnx_proto.TensorProto.FLOAT,
                              [1], [layer.iou_threshold])
    container.add_initializer(score_threshold, onnx_proto.TensorProto.FLOAT,
                              [1], [layer.score_threshold])

    container.add_node("NonMaxSuppression",
                       [box_batch, score_batch, max_output_size, iou_threshold, score_threshold],
                       operator.output_full_names + ['no_use'],
                       op_version=operator.target_opset, op_domain='com.microsoft')


set_converter(YOLONMSLayer, convert_NMSLayer)


def convert_model(yolo, name0, name1):
    yolo.load_model()
    if not os.path.exists(name0):
        onnxmodel = convert_keras(yolo.yolo_model, channel_first_inputs=['input_1'],
                                  debug_mode=True, custom_op_conversions=_custom_op_handlers)
        onnx.save_model(onnxmodel, name0)

    oxmlfinal = convert_keras(yolo.final_model, debug_mode=True, custom_op_conversions=_custom_op_handlers)
    onnx.save_model(oxmlfinal, name1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Need an image name to detect.")
        exit(-1)

    if '-c' in sys.argv:
        convert_model(YOLO(), 'model_data/yolov3_0.onnx', 'model_data/yolov3_1.onnx')
    else:
        detect_img(YOLO(), sys.argv[1])
