###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import sys
import unittest
import keras2onnx
import numpy as np
from keras2onnx.proto import keras
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import run_keras_and_ort, test_level_0
K = keras.backend

Activation = keras.layers.Activation
add = keras.layers.add
Average = keras.layers.Average
AveragePooling2D = keras.layers.AveragePooling2D
Add = keras.layers.Add
BatchNormalization = keras.layers.BatchNormalization
concatenate = keras.layers.concatenate
Conv1D = keras.layers.Conv1D
Conv2D = keras.layers.Conv2D
Conv3D = keras.layers.Conv3D
Convolution2D = keras.layers.Convolution2D
Convolution2DTranspose = keras.layers.Convolution2DTranspose
Dense = keras.layers.Dense
dot = keras.layers.dot
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Input = keras.layers.Input
Lambda = keras.layers.Lambda
LeakyReLU = keras.layers.LeakyReLU
MaxPool1D = keras.layers.MaxPool1D
MaxPooling2D = keras.layers.MaxPooling2D
multiply = keras.layers.multiply
Permute = keras.layers.Permute
Reshape = keras.layers.Reshape
SeparableConv2D = keras.layers.SeparableConv2D
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D

Sequential = keras.models.Sequential
Model = keras.models.Model


_image_scale_multiplier = 1
img_size = 128 * _image_scale_multiplier
stride = 64 * _image_scale_multiplier


class BaseSuperResolutionModel(object):

    def __init__(self, model_name, scale_factor):
        """
        Base model to provide a standard interface of adding Super Resolution models
        """
        self.model = None # type: Model
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.weight_path = None

        self.type_scale_type = "norm" # Default = "norm" = 1. / 255
        self.type_requires_divisible_shape = False
        self.type_true_upscaling = False

        self.evaluation_func = None
        self.uses_learning_phase = False

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128) -> Model:
        if self.type_requires_divisible_shape and height is not None and width is not None:
            assert height * _image_scale_multiplier % 4 == 0, "Height of the image must be divisible by 4"
            assert width * _image_scale_multiplier % 4 == 0, "Width of the image must be divisible by 4"

        if width is not None and height is not None:
            shape = (width * _image_scale_multiplier, height * _image_scale_multiplier, channels)
        else:
            shape = (None, None, channels)

        init = Input(shape=shape)

        return init


class ImageSuperResolutionModel(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ImageSuperResolutionModel, self).__init__("Image SR", scale_factor)

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/SR Weights %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        init = super(ImageSuperResolutionModel, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(self.n1, (self.f1, self.f1), activation='relu', padding='same', name='level1')(init)
        x = Convolution2D(self.n2, (self.f2, self.f2), activation='relu', padding='same', name='level2')(x)

        out = Convolution2D(channels, (self.f3, self.f3), padding='same', name='output')(x)

        model = Model(init, out)
        self.model = model
        return model


class ExpantionSuperResolution(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ExpantionSuperResolution, self).__init__("Expanded Image SR", scale_factor)

        self.f1 = 9
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/Expantion SR Weights %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        init = super(ExpantionSuperResolution, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(self.n1, (self.f1, self.f1), activation='relu', padding='same', name='level1')(init)

        x1 = Convolution2D(self.n2, (self.f2_1, self.f2_1), activation='relu', padding='same', name='lavel1_1')(x)
        x2 = Convolution2D(self.n2, (self.f2_2, self.f2_2), activation='relu', padding='same', name='lavel1_2')(x)
        x3 = Convolution2D(self.n2, (self.f2_3, self.f2_3), activation='relu', padding='same', name='lavel1_3')(x)

        x = Average()([x1, x2, x3])

        out = Convolution2D(channels, (self.f3, self.f3), activation='relu', padding='same', name='output')(x)

        model = Model(init, out)
        self.model = model
        return model


class DenoisingAutoEncoderSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(DenoisingAutoEncoderSR, self).__init__("Denoise AutoEncoder SR", scale_factor)

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/Denoising AutoEncoder %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to remove / reduce noise from upscaled images.
        """
        from keras.layers.convolutional import Deconvolution2D

        # Perform check that model input shape is divisible by 4
        init = super(DenoisingAutoEncoderSR, self).create_model(height, width, channels, load_weights, batch_size)

        output_shape = (None, width, height, channels)

        level1_1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(init)
        level2_1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(level1_1)

        level2_2 = Convolution2DTranspose(self.n1, (3, 3), activation='relu', padding='same')(level2_1)
        level2 = Add()([level2_1, level2_2])

        level1_2 = Convolution2DTranspose(self.n1, (3, 3), activation='relu', padding='same')(level2)
        level1 = Add()([level1_1, level1_2])

        decoded = Convolution2D(channels, (5, 5), activation='linear', padding='same')(level1)

        model = Model(init, decoded)

        self.model = model
        return model


class DeepDenoiseSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(DeepDenoiseSR, self).__init__("Deep Denoise SR", scale_factor)

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True

        self.n1 = 64
        self.n2 = 128
        self.n3 = 256

        self.weight_path = "weights/Deep Denoise Weights %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        # Perform check that model input shape is divisible by 4
        init = super(DeepDenoiseSR, self).create_model(height, width, channels, load_weights, batch_size)

        c1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(init)
        c1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(c1)

        x = MaxPooling2D((2, 2))(c1)

        c2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(x)
        c2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(c2)

        x = MaxPooling2D((2, 2))(c2)

        c3 = Convolution2D(self.n3, (3, 3), activation='relu', padding='same')(x)

        x = UpSampling2D()(c3)

        c2_2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(x)
        c2_2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(c2_2)

        m1 = Add()([c2, c2_2])
        m1 = UpSampling2D()(m1)

        c1_2 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(m1)
        c1_2 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(c1_2)

        m2 = Add()([c1, c1_2])

        decoded = Convolution2D(channels, 5, 5, activation='linear', border_mode='same')(m2)

        model = Model(init, decoded)
        self.model = model
        return model


class ResNetSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ResNetSR, self).__init__("ResNetSR", scale_factor)

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True
        self.uses_learning_phase = False

        self.n = 64
        self.mode = 2

        self.weight_path = "weights/ResNetSR %dX.h5" % (self.scale_factor)
        self.type_true_upscaling = True

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        init =  super(ResNetSR, self).create_model(height, width, channels, load_weights, batch_size)

        x0 = Convolution2D(64, (3, 3), activation='relu', padding='same', name='sr_res_conv1')(init)

        #x1 = Convolution2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2), name='sr_res_conv2')(x0)

        #x2 = Convolution2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2), name='sr_res_conv3')(x1)

        x = self._residual_block(x0, 1)

        nb_residual = 5
        for i in range(nb_residual):
            x = self._residual_block(x, i + 2)

        x = Add()([x, x0])

        x = self._upscale_block(x, 1)
        #x = Add()([x, x1])

        #x = self._upscale_block(x, 2)
        #x = Add()([x, x0])

        x = Convolution2D(3, (3, 3), activation="linear", padding='same', name='sr_res_conv_final')(x)

        model = Model(init, x)
        self.model = model
        return model

    def _residual_block(self, ip, id):
        mode = False if self.mode == 2 else None
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        init = ip

        x = Convolution2D(64, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_1')(ip)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_1")(x, training=mode)
        x = Activation('relu', name="sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(64, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_2")(x, training=mode)

        m = Add(name="sr_res_merge_" + str(id))([x, init])

        return m

    def _upscale_block(self, ip, id):
        init = ip

        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        channels = init._keras_shape[channel_dim]

        #x = Convolution2D(256, (3, 3), activation="relu", padding='same', name='sr_res_upconv1_%d' % id)(init)
        #x = SubPixelUpscaling(r=2, channels=self.n, name='sr_res_upscale1_%d' % id)(x)
        x = UpSampling2D()(init)
        x = Convolution2D(self.n, (3, 3), activation="relu", padding='same', name='sr_res_filter1_%d' % id)(x)

        # x = Convolution2DTranspose(channels, (4, 4), strides=(2, 2), padding='same', activation='relu',
        #                            name='upsampling_deconv_%d' % id)(init)

        return x


class GANImageSuperResolutionModel(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(GANImageSuperResolutionModel, self).__init__("GAN Image SR", scale_factor)

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

        self.gen_model = None # type: Model
        self.disc_model = None # type: Model

        self.type_scale_type = 'tanh'

        self.weight_path = "weights/GAN SR Weights %dX.h5" % (self.scale_factor)
        self.gen_weight_path = "weights/GAN SR Pretrain Weights %dX.h5" % (self.scale_factor)
        self.disc_weight_path = "weights/GAN SR Discriminator Weights %dX.h5" % (self.scale_factor)


    def create_model(self, mode='test', height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        channel_axis = -1

        gen_init = super(GANImageSuperResolutionModel, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(self.n1, (self.f1, self.f1), activation='relu', padding='same', name='gen_level1')(gen_init)
        x = LeakyReLU(alpha=0.25)(x)
        x = Convolution2D(self.n2, (self.f2, self.f2), activation='relu', padding='same', name='gen_level2')(x)
        x = LeakyReLU(alpha=0.25)(x)

        out = Convolution2D(channels, (self.f3, self.f3), activation='tanh', padding='same', name='gen_output')(x)

        gen_model = Model(gen_init, out)

        self.model = gen_model
        return self.model


class DistilledResNetSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(DistilledResNetSR, self).__init__("DistilledResNetSR", scale_factor)

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True
        self.uses_learning_phase = False

        self.n = 32
        self.mode = 2

        self.weight_path = "weights/DistilledResNetSR %dX.h5" % (self.scale_factor)
        self.type_true_upscaling = True

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        init =  super(DistilledResNetSR, self).create_model(height, width, channels, load_weights, batch_size)

        x0 = Convolution2D(self.n, (3, 3), activation='relu', padding='same', name='student_sr_res_conv1')(init)

        x = self._residual_block(x0, 1)

        x = Add(name='student_residual')([x, x0])
        x = self._upscale_block(x, 1)

        x = Convolution2D(3, (3, 3), activation="linear", padding='same', name='student_sr_res_conv_final')(x)

        model = Model(init, x)
        # dont compile yet
        if load_weights: model.load_weights(self.weight_path, by_name=True)

        self.model = model
        return model

    def _residual_block(self, ip, id):
        mode = False if self.mode == 2 else None
        channel_axis = -1
        init = ip

        x = Convolution2D(self.n, (3, 3), activation='linear', padding='same',
                          name='student_sr_res_conv_' + str(id) + '_1')(ip)
        x = BatchNormalization(axis=channel_axis, name="student_sr_res_batchnorm_" + str(id) + "_1")(x, training=mode)
        x = Activation('relu', name="student_sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(self.n, (3, 3), activation='linear', padding='same',
                          name='student_sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(axis=channel_axis, name="student_sr_res_batchnorm_" + str(id) + "_2")(x, training=mode)

        m = Add(name="student_sr_res_merge_" + str(id))([x, init])

        return m

    def _upscale_block(self, ip, id):
        init = ip

        channel_dim = -1
        channels = init._keras_shape[channel_dim]

        x = UpSampling2D(name='student_upsampling_%d' % id)(init)
        x = Convolution2D(self.n * 2, (3, 3), activation="relu", padding='same', name='student_sr_res_filter1_%d' % id)(x)

        return x


def _convND(ip, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, 1, padding='same', use_bias=False)(ip)
    elif rank == 4:
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False)(ip)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False)(ip)
    return x


def non_local_block(ip, computation_compression=2, mode='embedded'):
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    dim1, dim2, dim3 = None, None, None

    if len(ip_shape) == 3:  # time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # Video / Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, channels // 2)
        theta = Reshape((-1, channels // 2))(theta)

        # phi path
        phi = _convND(ip, rank, channels // 2)
        phi = Reshape((-1, channels // 2))(phi)

        f = dot([theta, phi], axes=2)

        # scale the values to make it size invariant
        if batchsize is not None:
            f = Lambda(lambda z: 1./ batchsize * z)(f)
        else:
            f = Lambda(lambda z: 1. / 128 * z)(f)


    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplemented('Concatenation mode has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, channels // 2)
        theta = Reshape((-1, channels // 2))(theta)

        # phi path
        phi = _convND(ip, rank, channels // 2)
        phi = Reshape((-1, channels // 2))(phi)

        if computation_compression > 1:
            # shielded computation
            phi = MaxPool1D(computation_compression)(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)

    # g path
    g = _convND(ip, rank, channels // 2)
    g = Reshape((-1, channels // 2))(g)

    if computation_compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(computation_compression)(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, channels // 2))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, channels // 2))(y)
        else:
            y = Reshape((channels // 2, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, channels // 2))(y)
        else:
            y = Reshape((channels // 2, dim1, dim2, dim3))(y)

    # project filters
    y = _convND(y, rank, channels)

    # residual connection
    residual = add([ip, y])

    return residual


class NonLocalResNetSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(NonLocalResNetSR, self).__init__("NonLocalResNetSR", scale_factor)

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True
        self.uses_learning_phase = False

        self.n = 32
        self.mode = 2

        self.weight_path = "weights/NonLocalResNetSR %dX.h5" % (self.scale_factor)
        self.type_true_upscaling = True

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        init =  super(NonLocalResNetSR, self).create_model(height, width, channels, load_weights, batch_size)

        x0 = Convolution2D(self.n, (3, 3), activation='relu', padding='same', name='sr_res_conv1')(init)
        x0 = non_local_block(x0)

        x = self._residual_block(x0, 1)

        nb_residual = 5
        for i in range(nb_residual):
            x = self._residual_block(x, i + 2)

        x = non_local_block(x, computation_compression=2)
        x = Add()([x, x0])

        x = self._upscale_block(x, 1)

        x = Convolution2D(3, (3, 3), activation="linear", padding='same', name='sr_res_conv_final')(x)

        model = Model(init, x)
        self.model = model
        return model

    def _residual_block(self, ip, id):
        mode = False if self.mode == 2 else None
        channel_axis = -1
        init = ip

        x = Convolution2D(self.n, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_1')(ip)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_1")(x, training=mode)
        x = Activation('relu', name="sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(self.n, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_2")(x, training=mode)

        m = Add(name="sr_res_merge_" + str(id))([x, init])

        return m

    def _upscale_block(self, ip, id):
        init = ip

        channel_dim = -1

        x = UpSampling2D()(init)
        x = Convolution2D(self.n, (3, 3), activation="relu", padding='same', name='sr_res_filter1_%d' % id)(x)

        return x


# Model from https://github.com/titu1994/Image-Super-Resolution
class TestSuperResolution(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_ImageSuperResolutionModel(self):
        K.clear_session()
        model_type = ImageSuperResolutionModel(2.0)
        keras_model = model_type.create_model()
        data = np.random.rand(2, 32, 32, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    def test_ExpantionSuperResolution(self):
        K.clear_session()
        model_type = ExpantionSuperResolution(2.0)
        keras_model = model_type.create_model()
        data = np.random.rand(2, 32, 32, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_DenoisingAutoEncoderSR(self):
        K.clear_session()
        model_type = DenoisingAutoEncoderSR(2.0)
        keras_model = model_type.create_model()
        data = np.random.rand(2, 32, 32, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_DeepDenoiseSR(self):
        K.clear_session()
        model_type = DeepDenoiseSR(2.0)
        keras_model = model_type.create_model()
        data = np.random.rand(2, 32, 32, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_ResNetSR(self):
        K.clear_session()
        model_type = ResNetSR(2.0)
        keras_model = model_type.create_model()
        data = np.random.rand(2, 32, 32, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_GANImageSuperResolutionModel(self):
        K.clear_session()
        model_type = GANImageSuperResolutionModel(2.0)
        keras_model = model_type.create_model()
        data = np.random.rand(2, 32, 32, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_DistilledResNetSR(self):
        K.clear_session()
        model_type = DistilledResNetSR(2.0)
        keras_model = model_type.create_model()
        data = np.random.rand(2, 32, 32, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_NonLocalResNetSR(self):
        K.clear_session()
        model_type = NonLocalResNetSR(2.0)
        keras_model = model_type.create_model()
        data = np.random.rand(2, 32, 32, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
