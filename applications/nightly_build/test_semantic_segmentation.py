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
import tensorflow as tf

Activation = keras.layers.Activation
AveragePooling2D = keras.layers.AveragePooling2D
add = keras.layers.add
Add = keras.layers.Add
BatchNormalization = keras.layers.BatchNormalization
concatenate = keras.layers.concatenate
Concatenate = keras.layers.Concatenate
Conv2D = keras.layers.Conv2D
Conv2DTranspose = keras.layers.Conv2DTranspose
DepthwiseConv2D = keras.layers.DepthwiseConv2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Input = keras.layers.Input
Lambda = keras.layers.Lambda
LeakyReLU = keras.layers.LeakyReLU
MaxPooling2D = keras.layers.MaxPooling2D
multiply = keras.layers.multiply
Permute = keras.layers.Permute
PReLU = keras.layers.PReLU
Reshape = keras.layers.Reshape
SeparableConv2D = keras.layers.SeparableConv2D
SpatialDropout2D = keras.layers.SpatialDropout2D
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D

Sequential = keras.models.Sequential
Model = keras.models.Model


def ConvBNLayer(x, out_channels, kernel_size, stride=1, dilation=1, act=True):
    x = Conv2D(out_channels, kernel_size, strides=stride,
               padding='same', dilation_rate=dilation)(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    if act:
        return Activation('relu')(x)
    else:
        return x


def ACBlock(x, out_channels, kernel_size, stride=1, padding=0, dilation=1,
            groups=1, deploy=False):
    if deploy:
        return Conv2D(out_channels, (kernel_size, kernel_size), strides=stride,
                      dilation_rate=dilation, use_bias=True, padding='same')(x)
    else:
        square_outputs = Conv2D(out_channels, (kernel_size, kernel_size), strides=stride,
                                dilation_rate=dilation, use_bias=False, padding='same')(x)
        square_outputs = BatchNormalization(epsilon=1e-5, momentum=0.1)(square_outputs)

        center_offset_from_origin_border = padding - kernel_size // 2
        ver_pad_or_crop = (padding, center_offset_from_origin_border)
        hor_pad_or_crop = (center_offset_from_origin_border, padding)

        if center_offset_from_origin_border >= 0:
            vertical_outputs = x
            ver_conv_padding = ver_pad_or_crop
            horizontal_outputs = x
            hor_conv_padding = hor_pad_or_crop
        else:
            vertical_outputs = x

            ver_conv_padding = (0, 0)
            horizontal_outputs = x

            hor_conv_padding = (0, 0)

        vertical_outputs = ZeroPadding2D(padding=ver_conv_padding)(vertical_outputs)
        vertical_outputs = Conv2D(out_channels, kernel_size=(kernel_size, 1),
                                  strides=stride, padding='same', use_bias=False,
                                  dilation_rate=dilation)(vertical_outputs)
        vertical_outputs = BatchNormalization(epsilon=1e-5, momentum=0.1)(vertical_outputs)

        horizontal_outputs = ZeroPadding2D(padding=hor_conv_padding)(horizontal_outputs)
        horizontal_outputs = Conv2D(out_channels, kernel_size=(kernel_size, 1),
                                    strides=stride, padding='same', use_bias=False,
                                    dilation_rate=dilation)(horizontal_outputs)
        horizontal_outputs = BatchNormalization(epsilon=1e-5, momentum=0.1)(horizontal_outputs)

        results = Add()([square_outputs, vertical_outputs, horizontal_outputs])

        return results


def BasicBlock(x, out_channels, stride=1, downsample=False):
    residual = x
    x = ACBlock(x, out_channels, kernel_size=3, stride=stride)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = Activation('relu')(x)
    x = ACBlock(x, out_channels, kernel_size=3)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)

    if downsample:
        shortcut = ConvBNLayer(residual, out_channels, kernel_size=1, stride=stride)
        outputs = Add()([x, shortcut])
    else:
        outputs = Add()([x, residual])

    return Activation('relu')(outputs)


def BottleNeckBlock(x, out_channels, stride=1, downsample=False):
    expansion = 4

    residual = x

    x = ConvBNLayer(x, out_channels, kernel_size=1, act=True)

    x = ACBlock(x, out_channels, kernel_size=3, stride=stride)

    x = ConvBNLayer(x, out_channels * expansion, kernel_size=1, act=False)

    if downsample:
        shortcut_tensor = ConvBNLayer(residual, out_channels * 4, kernel_size=1, stride=stride)
    else:
        shortcut_tensor = residual

    outputs = Add()([x, shortcut_tensor])
    return Activation('relu')(outputs)


def ResNet(x, block_type, layers_repeat, class_dim=1000):
    num_filters = [64, 128, 256, 512]

    x = ConvBNLayer(x, 64, kernel_size=7, stride=2, act=True)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    for block in range(4):
        downsample = True
        for i in range(layers_repeat[block]):
            x = block_type(x, num_filters[block], stride=2 if i == 0 and block != 0 else 1, downsample=downsample)
            downsample = False

    pool = GlobalAveragePooling2D()(x)
    output = Dense(class_dim, activation='relu')(pool)
    return output


def ResACNet(x, class_dim=1000, depth=50):
    assert depth in [10, 18, 34, 50, 101, 152, 200]

    if depth == 10:
        output = ResNet(x, BasicBlock, [1, 1, 1, 1], class_dim)
    elif depth == 18:
        output = ResNet(x, BasicBlock, [2, 2, 2, 2], class_dim)
    elif depth == 34:
        output = ResNet(x, BasicBlock, [3, 4, 6, 3], class_dim)
    elif depth == 50:
        output = ResNet(x, BottleNeckBlock, [3, 4, 6, 3], class_dim)
    elif depth == 101:
        output = ResNet(x, BottleNeckBlock, [3, 4, 23, 3], class_dim)
    elif depth == 152:
        output = ResNet(x, BottleNeckBlock, [3, 8, 36, 3], class_dim)
    elif depth == 200:
        output = ResNet(x, BottleNeckBlock, [3, 24, 36, 3], class_dim)
    return output


def conv_block(input, filters):
    out = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def up_conv(input, filters):
    out = UpSampling2D()(input)
    out = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def Attention_block(input1, input2, filters):
    g1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input1)
    g1 = BatchNormalization()(g1)
    x1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input2)
    x1 = BatchNormalization()(x1)
    psi = Activation('relu')(add([g1, x1]))
    psi = Conv2D(filters, kernel_size=1, strides=1, padding='same')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    out = multiply([input2, psi])
    return out


def AttUNet(nClasses, input_height=224, input_width=224):
    inputs = Input(shape=(input_height, input_width, 3))
    n1 = 32
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    e1 = conv_block(inputs, filters[0])

    e2 = MaxPooling2D(strides=2)(e1)
    e2 = conv_block(e2, filters[1])

    e3 = MaxPooling2D(strides=2)(e2)
    e3 = conv_block(e3, filters[2])

    e4 = MaxPooling2D(strides=2)(e3)
    e4 = conv_block(e4, filters[3])

    e5 = MaxPooling2D(strides=2)(e4)
    e5 = conv_block(e5, filters[4])

    d5 = up_conv(e5, filters[3])
    x4 = Attention_block(d5, e4, filters[3])
    d5 = Concatenate()([x4, d5])
    d5 = conv_block(d5, filters[3])

    d4 = up_conv(d5, filters[2])
    x3 = Attention_block(d4, e3, filters[2])
    d4 = Concatenate()([x3, d4])
    d4 = conv_block(d4, filters[2])

    d3 = up_conv(d4, filters[1])
    x2 = Attention_block(d3, e2, filters[1])
    d3 = Concatenate()([x2, d3])
    d3 = conv_block(d3, filters[1])

    d2 = up_conv(d3, filters[0])
    x1 = Attention_block(d2, e1, filters[0])
    d2 = Concatenate()([x1, d2])
    d2 = conv_block(d2, filters[0])

    o = Conv2D(nClasses, (3, 3), padding='same')(d2)

    outputHeight = Model(inputs, o).output_shape[1]
    outputWidth = Model(inputs, o).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(o)
    out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model


from keras.backend.common import normalize_data_format

class BilinearUpsampling(keras.layers.Layer):

    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.upsampling = keras.utils.conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = keras.layers.InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        # .tf
        return tf.image.resize_bilinear(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                 int(inputs.shape[2] * self.upsampling[1])))

    def get_config(self):

        config = {'size': self.upsampling, 'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def xception_downsample_block(x, channels, top_relu=False):
    ##separable conv1
    if top_relu:
        x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ##separable conv2
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ##separable conv3
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def res_xception_downsample_block(x, channels):
    res = Conv2D(channels, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = xception_downsample_block(x, channels)
    x = add([x, res])
    return x


def xception_block(x, channels):
    ##separable conv1
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    ##separable conv2
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    ##separable conv3
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def res_xception_block(x, channels):
    res = x
    x = xception_block(x, channels)
    x = add([x, res])
    return x


def aspp(x, input_shape, out_stride):
    b0 = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    b0 = BatchNormalization()(b0)
    b0 = Activation("relu")(b0)

    b1 = DepthwiseConv2D((3, 3), dilation_rate=(6, 6), padding="same", use_bias=False)(x)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)
    b1 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)

    b2 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)
    b2 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)

    b3 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)
    b3 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)

    out_shape = int(input_shape[0] / out_stride)
    b4 = AveragePooling2D(pool_size=(out_shape, out_shape))(x)
    b4 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b4)
    b4 = BatchNormalization()(b4)
    b4 = Activation("relu")(b4)
    b4 = BilinearUpsampling((out_shape, out_shape))(b4)

    x = Concatenate()([b4, b0, b1, b2, b3])
    return x


def DeeplabV3_plus(nClasses=21, input_height=512, input_width=512, out_stride=16):
    img_input = Input(shape=(input_height, input_width, 3))
    x = Conv2D(32, (3, 3), strides=(2, 2), padding="same", use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = res_xception_downsample_block(x, 128)

    res = Conv2D(256, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    skip = BatchNormalization()(x)
    x = Activation("relu")(skip)
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = add([x, res])

    x = xception_downsample_block(x, 728, top_relu=True)

    for i in range(16):
        x = res_xception_block(x, 728)

    res = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(728, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = add([x, res])

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1536, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1536, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(2048, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # aspp
    x = aspp(x, (input_height, input_width, 3), out_stride)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.9)(x)

    ##decoder
    x = BilinearUpsampling((4, 4))(x)
    dec_skip = Conv2D(48, (1, 1), padding="same", use_bias=False)(skip)
    dec_skip = BatchNormalization()(dec_skip)
    dec_skip = Activation("relu")(dec_skip)
    x = Concatenate()([x, dec_skip])

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(nClasses, (1, 1), padding="same")(x)
    x = BilinearUpsampling((4, 4))(x)
    outputHeight = Model(img_input, x).output_shape[1]
    outputWidth = Model(img_input, x).output_shape[2]
    x = (Reshape((outputHeight * outputWidth, nClasses)))(x)
    x = Activation('softmax')(x)
    model = Model(input=img_input, output=x)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth
    return model


def initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, strides=(2, 2)):
    conv = Conv2D(nb_filter, (nb_row, nb_col), padding='same', strides=strides)(inp)
    max_pool = MaxPooling2D()(inp)
    merged = concatenate([conv, max_pool], axis=3)
    return merged

def bottleneck(inp, output, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = output // internal_scale
    encoder = inp
    # 1x1
    input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = Conv2D(internal, (input_stride, input_stride),
                     # padding='same',
                     strides=(input_stride, input_stride), use_bias=False)(encoder)
    # Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # conv
    if not asymmetric and not dilated:
        encoder = Conv2D(internal, (3, 3), padding='same')(encoder)
    elif asymmetric:
        encoder = Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(encoder)
        encoder = Conv2D(internal, (asymmetric, 1), padding='same')(encoder)
    elif dilated:
        encoder = Conv2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding='same')(encoder)
    else:
        raise (Exception('You shouldn\'t be here'))

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # 1x1
    encoder = Conv2D(output, (1, 1), use_bias=False)(encoder)

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)

    other = inp
    # other branch
    if downsample:
        other = MaxPooling2D()(other)

        other = Permute((1, 3, 2))(other)
        pad_feature_maps = output - inp.get_shape().as_list()[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = Permute((1, 3, 2))(other)

    encoder = add([encoder, other])
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    return encoder

def en_build(inp, dropout_rate=0.01):
    enet = initial_block(inp)
    enet = BatchNormalization(momentum=0.1)(enet)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
    enet = PReLU(shared_axes=[1, 2])(enet)
    enet = bottleneck(enet, 64, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
    for _ in range(4):
        enet = bottleneck(enet, 64, dropout_rate=dropout_rate)  # bottleneck 1.i

    enet = bottleneck(enet, 128, downsample=True)  # bottleneck 2.0
    # bottleneck 2.x and 3.x
    for _ in range(2):
        enet = bottleneck(enet, 128)  # bottleneck 2.1
        enet = bottleneck(enet, 128, dilated=2)  # bottleneck 2.2
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.3
        enet = bottleneck(enet, 128, dilated=4)  # bottleneck 2.4
        enet = bottleneck(enet, 128)  # bottleneck 2.5
        enet = bottleneck(enet, 128, dilated=8)  # bottleneck 2.6
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.7
        enet = bottleneck(enet, 128, dilated=16)  # bottleneck 2.8

    return enet

# decoder
def de_bottleneck(encoder, output, upsample=False, reverse_module=False):
    internal = output // 4

    x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    if not upsample:
        x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(output, (1, 1), padding='same', use_bias=False)(x)

    other = encoder
    if encoder.get_shape()[-1] != output or upsample:
        other = Conv2D(output, (1, 1), padding='same', use_bias=False)(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module is not False:
            other = UpSampling2D(size=(2, 2))(other)

    if upsample and reverse_module is False:
        decoder = x
    else:
        x = BatchNormalization(momentum=0.1)(x)
        decoder = add([x, other])
        decoder = Activation('relu')(decoder)

    return decoder

def de_build(encoder, nc):
    enet = de_bottleneck(encoder, 64, upsample=True, reverse_module=True)  # bottleneck 4.0
    enet = de_bottleneck(enet, 64)  # bottleneck 4.1
    enet = de_bottleneck(enet, 64)  # bottleneck 4.2
    enet = de_bottleneck(enet, 16, upsample=True, reverse_module=True)  # bottleneck 5.0
    enet = de_bottleneck(enet, 16)  # bottleneck 5.1

    enet = Conv2DTranspose(filters=nc, kernel_size=(2, 2), strides=(2, 2), padding='same')(enet)
    return enet

def ENet(n_classes, input_height=256, input_width=256):
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    img_input = Input(shape=(input_height, input_width, 3))
    enet = en_build(img_input)
    enet = de_build(enet, n_classes)
    o_shape = Model(img_input, enet).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    enet = (Reshape((outputHeight*outputWidth, n_classes)))(enet)
    enet = Activation('softmax')(enet)
    model = Model(img_input, enet)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model


# Model from https://github.com/BBuf/Keras-Semantic-Segmentation
class TestSemanticSegmentation(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_ResACNet(self):
        K.clear_session()
        input_shape = (224, 224, 3)
        inputs = Input(shape=input_shape, name="inputs")
        y = ResACNet(inputs, depth=50)
        keras_model = keras.models.Model(inputs=inputs, outputs=y)
        data = np.random.rand(2, 224, 224, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files, compare_perf=True))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_AttUNet(self):
        K.clear_session()
        keras_model = AttUNet(80)
        data = np.random.rand(2, 224, 224, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files, compare_perf=True))


    @unittest.skipIf(test_level_0,
                     "Test level 0 only.")
    def test_DeepLabV3Plus(self):
        K.clear_session()
        keras_model = DeeplabV3_plus(input_height=224, input_width=224)
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files, compare_perf=True))


    def test_ENet(self):
        K.clear_session()
        keras_model = ENet(80)
        data = np.random.rand(1, 256, 256, 3).astype(np.float32)
        expected = keras_model.predict(data)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, keras_model, data, expected, self.model_files, compare_perf=True))


if __name__ == "__main__":
    unittest.main()
