import unittest
import keras2onnx
import numpy as np
import tensorflow as tf
from test_utils import run_onnx_runtime


class LeNet(tf.keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=6,
                                               kernel_size=(3, 3), activation='relu',
                                               input_shape=(32, 32, 1))
        self.average_pool = tf.keras.layers.AveragePooling2D()
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=16,
                                               kernel_size=(3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc_1 = tf.keras.layers.Dense(120, activation='relu')
        self.fc_2 = tf.keras.layers.Dense(84, activation='relu')
        self.out = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, input):
        x = self.conv2d_1(input)
        x = self.average_pool(x)
        x = self.conv2d_2(x)
        x = self.average_pool(x)
        x = self.flatten(x)
        x = self.fc_2(self.fc_1(x))
        return self.out(x)


@unittest.skipIf(not keras2onnx.proto.tfcompat.is_tf2, "Tensorflow 2.0 only tests.")
class TestTF2Keras2ONNX(unittest.TestCase):
    def test_tf2_keras_model(self):
        lenet = LeNet()
        data = np.random.rand(2 * 416 * 416 * 3).astype(np.float32).reshape(2, 416, 416, 3)
        expected = lenet(data)
        lenet._set_inputs(data)
        oxml = keras2onnx.convert_keras(lenet, debug_mode=True)
        model_files = []
        self.assertTrue(run_onnx_runtime('lenet', oxml, data, expected, model_files))


if __name__ == "__main__":
    unittest.main()
