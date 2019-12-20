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


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output


@unittest.skipIf((not keras2onnx.proto.is_tf_keras) or (not keras2onnx.proto.tfcompat.is_tf2),
                 "Tensorflow 2.0 only tests.")
class TestTF2Keras2ONNX(unittest.TestCase):
    def test_lenet(self):
        tf.keras.backend.clear_session()
        lenet = LeNet()
        data = np.random.rand(2 * 416 * 416 * 3).astype(np.float32).reshape(2, 416, 416, 3)
        expected = lenet(data)
        lenet._set_inputs(data)
        oxml = keras2onnx.convert_keras(lenet)
        model_files = []
        self.assertTrue(run_onnx_runtime('lenet', oxml, data, expected, model_files))

    def test_mlf(self):
        tf.keras.backend.clear_session()
        mlf = MLP()
        input = tf.random.normal((2, 20))
        expected = mlf(input)
        mlf._set_inputs(input)
        oxml = keras2onnx.convert_keras(mlf)
        model_files = []
        self.assertTrue(run_onnx_runtime('lenet', oxml, input.numpy(), expected, model_files))


if __name__ == "__main__":
    unittest.main()
