###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import pytest
import keras2onnx
import numpy as np
import tensorflow as tf

if (not keras2onnx.proto.is_tf_keras) or (not keras2onnx.proto.tfcompat.is_tf2):
    pytest.skip("Tensorflow 2.0 only tests.", allow_module_level=True)


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


class SimpleWrapperModel(tf.keras.Model):
    def __init__(self, func):
        super(SimpleWrapperModel, self).__init__()
        self.func = func

    def call(self, inputs, **kwargs):
        return self.func(inputs)


def test_lenet(runner):
    tf.keras.backend.clear_session()
    lenet = LeNet()
    data = np.random.rand(2 * 416 * 416 * 3).astype(np.float32).reshape(2, 416, 416, 3)
    expected = lenet(data)
    lenet._set_inputs(data)
    oxml = keras2onnx.convert_keras(lenet)
    assert runner('lenet', oxml, data, expected)


def test_mlf(runner):
    tf.keras.backend.clear_session()
    mlf = MLP()
    np_input = tf.random.normal((2, 20))
    expected = mlf.predict(np_input)
    oxml = keras2onnx.convert_keras(mlf)
    assert runner('lenet', oxml, np_input.numpy(), expected)


def test_tf_ops(runner):
    tf.keras.backend.clear_session()

    def op_func(arg_inputs):
        x = tf.math.squared_difference(arg_inputs[0], arg_inputs[1])
        x = tf.matmul(x, x, adjoint_b=True)
        r = tf.rank(x)
        x = x - tf.cast(tf.expand_dims(r, axis=0), tf.float32)
        return x

    dm = SimpleWrapperModel(op_func)
    inputs = [tf.random.normal((3, 2, 20)), tf.random.normal((3, 2, 20))]
    expected = dm.predict(inputs)
    oxml = keras2onnx.convert_keras(dm)
    assert runner('op_model', oxml, [i_.numpy() for i_ in inputs], expected)


layers = tf.keras.layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # epsilon = tf.fill(dims=(batch, dim), value=.9)
        # epsilon = tf.compat.v1.random_normal(shape=(batch, dim), seed=1234)
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=12340)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self,
                 latent_dim=32,
                 intermediate_dim=64,
                 name='encoder',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self,
                 original_dim,
                 intermediate_dim=64,
                 name='decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = layers.Dense(original_dim, activation='sigmoid')

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 original_dim,
                 intermediate_dim=64,
                 latent_dim=32,
                 name='autoencoder',
                 **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed


def test_auto_encoder(runner):
    tf.keras.backend.clear_session()
    original_dim = 20
    vae = VariationalAutoEncoder(original_dim, 64, 32)
    x = tf.random.normal((7, original_dim))
    expected = vae.predict(x)
    oxml = keras2onnx.convert_keras(vae)
    # assert runner('variational_auto_encoder', oxml, [x.numpy()], expected)
    # The random generator is not same between different engiens.
    import onnx
    onnx.checker.check_model(oxml)


def test_tf_where(runner):
    def _tf_where(input_0):
        a = tf.where(True, input_0, [0, 1, 2, 5, 7])
        b = tf.where([True], tf.expand_dims(input_0, axis=0), tf.expand_dims([0, 1, 2, 5, 7], axis=0))
        c = tf.logical_or(tf.cast(a, tf.bool), tf.cast(b, tf.bool))
        return c

    swm = SimpleWrapperModel(_tf_where)
    const_in = [np.array([2, 4, 6, 8, 10])]
    expected = swm(const_in)
    swm._set_inputs(const_in)
    oxml = keras2onnx.convert_keras(swm, debug_mode=True)
    assert runner('where_test', oxml, const_in, expected)
