import os
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

    def call(self, inputs, **kwargs):
        x = self.conv2d_1(inputs)
        x = self.average_pool(x)
        x = self.conv2d_2(x)
        x = self.average_pool(x)
        x = self.flatten(x)
        x = self.fc_2(self.fc_1(x))
        return self.out(x)


class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs, **kwargs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output


class DummyModel(tf.keras.Model):
    def __init__(self, func):
        super(DummyModel, self).__init__()
        self.func = func

    def call(self, inputs, **kwargs):
        return self.func(inputs)


@unittest.skipIf((not keras2onnx.proto.is_tf_keras) or (not keras2onnx.proto.tfcompat.is_tf2),
                 "Tensorflow 2.0 only tests.")
class TestTF2Keras2ONNX(unittest.TestCase):
    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def test_lenet(self):
        tf.keras.backend.clear_session()
        lenet = LeNet()
        data = np.random.rand(2 * 416 * 416 * 3).astype(np.float32).reshape(2, 416, 416, 3)
        expected = lenet(data)
        lenet._set_inputs(data)
        oxml = keras2onnx.convert_keras(lenet)
        self.assertTrue(run_onnx_runtime('lenet', oxml, data, expected, self.model_files))

    def test_mlf(self):
        tf.keras.backend.clear_session()
        mlf = MLP()
        np_input = tf.random.normal((2, 20))
        expected = mlf.predict(np_input)
        oxml = keras2onnx.convert_keras(mlf)
        self.assertTrue(run_onnx_runtime('lenet', oxml, np_input.numpy(), expected, self.model_files))

    def test_tf_ops(self):
        tf.keras.backend.clear_session()

        def op_func(arg_inputs):
            x = tf.math.squared_difference(arg_inputs[0], arg_inputs[1])
            x = tf.matmul(x, x, adjoint_b=True)
            return x

        dm = DummyModel(op_func)
        inputs = [tf.random.normal((3, 2, 20)), tf.random.normal((3, 2, 20))]
        expected = dm.predict(inputs)
        oxml = keras2onnx.convert_keras(dm)
        self.assertTrue(run_onnx_runtime('op_model', oxml, [i_.numpy() for i_ in inputs], expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
