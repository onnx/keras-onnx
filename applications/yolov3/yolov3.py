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
from keras2onnx.common.onnx_ops import apply_transpose, apply_identity
from keras2onnx.proto import onnx_proto
from custom_ops import on_StridedSlice, on_Round

import yolo3
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body, yolo_boxes_and_scores
from yolo3.utils import letterbox_image


class YOLOEvaluationLayer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(YOLOEvaluationLayer, self).__init__()
        self.anchors = np.array(kwargs.get('anchors'))
        self.num_classes = kwargs.get('num_classes')

    def get_config(self):
        config = {
            "anchors": self.anchors,
            "num_classes": self.num_classes,
        }

        return config

    def call(self, inputs, **kwargs):
        """Evaluate YOLO model on given input and return filtered boxes."""
        yolo_outputs = inputs[0:3]
        input_image_shape = K.squeeze(inputs[3], axis=0)
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
        return [(None, 4), (None, None)]


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
        return [boxes_r, scores_r, classes_]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return [(None, None, 4), (None, None, None), (None, None)]


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

        image_input = keras.Input(shape=(None, None, 3), dtype='float32')
        backbone = keras.models.clone_model(self.yolo_model)
        y1, y2, y3 = backbone([image_input])

        input_image_shape = keras.Input(shape=(2,), dtype='int32', name='image_shape')
        boxes, box_scores = \
            YOLOEvaluationLayer(anchors=self.anchors, num_classes=len(self.class_names))(
                inputs=[y1, y2, y3, input_image_shape])

        out_boxes, out_scores, out_indices = \
            YOLONMSLayer(anchors=self.anchors, num_classes=len(self.class_names))(
                inputs=[boxes, box_scores])
        self.final_model = keras.Model(inputs=[image_input, input_image_shape],
                                       outputs=[out_boxes, out_scores, out_indices])

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
        image_data /= 255.
        image_data = np.transpose(image_data, [2, 0, 1])

        print(image_data.shape)
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        r = self.session.run(None, input_feed={'input_1:01': image_data})
        feed_f = dict(zip(['image_shape:01', 'y1:01', 'y2:01', 'y3:01'],
                          (np.array([image.size[1], image.size[0]], dtype=np.int32).reshape(1, 2),
                           r[0],
                           r[1],
                           r[2])))
        all_boxes, all_scores, indices = self.session_final.run(None, input_feed=feed_f)

        out_boxes, out_scores, out_classes = [], [], []
        for idx_ in indices:
            out_classes.append(idx_[1])
            out_scores.append(all_scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(all_boxes[idx_1])

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

        font = ImageFont.truetype(font=self._get_data_path('font/FiraMono-Medium.otf'),
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

    # apply_transpose(scope, operator.inputs[0].full_name, box_transpose, container, perm=[2, 0, 1])
    apply_identity(scope, operator.inputs[0].full_name, box_transpose, container)
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
                              [], [layer.max_boxes])
    container.add_initializer(iou_threshold, onnx_proto.TensorProto.FLOAT,
                              [], [layer.iou_threshold])
    container.add_initializer(score_threshold, onnx_proto.TensorProto.FLOAT,
                              [], [layer.score_threshold])

    nms_node = next((nd_ for nd_ in operator.node_list if nd_.type == 'NonMaxSuppressionV3'), operator.node_list[0])
    container.add_node("NonMaxSuppression",
                       [box_batch, score_batch, max_output_size, iou_threshold, score_threshold],
                       operator.output_full_names[2],
                       op_version=operator.target_opset, op_domain='com.microsoft',
                       name=nms_node.name)

    apply_identity(scope, box_batch, operator.output_full_names[0], container)
    apply_identity(scope, score_batch, operator.output_full_names[1], container)


set_converter(YOLONMSLayer, convert_NMSLayer)


def convert_model(yolo, name):
    # yolo.load_model()
    target_opset_number = 10
    kmlfinal = keras.models.load_model('model_data/final_model.h5',
                                       custom_objects={'YOLOEvaluationLayer': YOLOEvaluationLayer,
                                                       'YOLONMSLayer': YOLONMSLayer})
    oxmlfinal = convert_keras(kmlfinal, debug_mode=True,
                              custom_op_conversions=_custom_op_handlers, target_opset=target_opset_number)
    onnx.save_model(oxmlfinal, name)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Need an image name to detect.")
        exit(-1)

    if '-c' in sys.argv:
        convert_model(YOLO(), 'model_data/yolov3.onnx')
    else:
        detect_img(YOLO(), sys.argv[1])
