###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import sys
import onnx
import unittest
import keras2onnx
import keras_segmentation
import numpy as np
from keras2onnx.proto import keras, is_keras_older_than
from distutils.version import StrictVersion
from os.path import dirname, abspath

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import run_image, run_onnx_runtime
img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')

Activation = keras.layers.Activation
AveragePooling2D = keras.layers.AveragePooling2D
BatchNormalization = keras.layers.BatchNormalization
Bidirectional = keras.layers.Bidirectional
Concatenate = keras.layers.Concatenate
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
Input = keras.layers.Input
LeakyReLU = keras.layers.LeakyReLU
LSTM = keras.layers.LSTM
MaxPooling2D = keras.layers.MaxPooling2D
multiply = keras.layers.multiply
Reshape = keras.layers.Reshape
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D

Sequential = keras.models.Sequential

class TestKerasApplications(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def test_MobileNet(self):
        mobilenet = keras.applications.mobilenet
        model = mobilenet.MobileNet(weights='imagenet')
        res = run_image(model, self.model_files, img_path)
        self.assertTrue(*res)

    @unittest.skipIf(is_keras_older_than("2.2.3"),
                     "There is no mobilenet_v2 module before keras 2.2.3.")
    def test_MobileNetV2(self):
        mobilenet_v2 = keras.applications.mobilenet_v2
        model = mobilenet_v2.MobileNetV2(weights='imagenet')
        res = run_image(model, self.model_files, img_path)
        self.assertTrue(*res)

    def test_ResNet50(self):
        from keras.applications.resnet50 import ResNet50
        model = ResNet50(include_top=True, weights='imagenet')
        res = run_image(model, self.model_files, img_path)
        self.assertTrue(*res)

    def test_InceptionV3(self):
        from keras.applications.inception_v3 import InceptionV3
        model = InceptionV3(include_top=True, weights='imagenet')
        res = run_image(model, self.model_files, img_path, target_size=299)
        self.assertTrue(*res)

    def test_DenseNet121(self):
        from keras.applications.densenet import DenseNet121
        model = DenseNet121(include_top=True, weights='imagenet')
        res = run_image(model, self.model_files, img_path)
        self.assertTrue(*res)

    def test_Xception(self):
        from keras.applications.xception import Xception
        model = Xception(include_top=True, weights='imagenet')
        res = run_image(model, self.model_files, img_path, atol=5e-3, target_size=299)
        self.assertTrue(*res)

    def test_SmileCNN(self):
        # From https://github.com/kylemcdonald/SmileCNN/blob/master/2%20Training.ipynb
        nb_filters = 32
        nb_pool = 2
        nb_conv = 3
        nb_classes = 2

        model = Sequential()

        model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu', input_shape=(32, 32, 3)))
        model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))
        res = run_image(model, self.model_files, img_path, atol=5e-3, target_size=32)
        self.assertTrue(*res)

    @unittest.skipIf(is_keras_older_than("2.2.4"),
                     "keras-resnet requires keras 2.2.4 or later.")
    def test_keras_resnet_batchnormalization(self):
        N, C, H, W = 2, 3, 120, 120
        import keras_resnet

        model = Sequential()
        model.add(ZeroPadding2D(padding=((3, 3), (3, 3)), input_shape=(H, W, C), data_format='channels_last'))
        model.add(Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=False,
                         data_format='channels_last'))
        model.add(keras_resnet.layers.BatchNormalization(freeze=True, axis=3))

        onnx_model = keras2onnx.convert_keras(model, model.name)
        data = np.random.rand(N, H, W, C).astype(np.float32).reshape((N, H, W, C))
        expected = model.predict(data)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    def test_tcn(self):
        from tcn import TCN
        batch_size, timesteps, input_dim = None, 20, 1
        actual_batch_size = 3
        i = Input(batch_shape=(batch_size, timesteps, input_dim))
        np.random.seed(1000)  # set the random seed to avoid the output result discrepancies.
        for return_sequences in [True, False]:
            o = TCN(return_sequences=return_sequences)(i)  # The TCN layers are here.
            o = Dense(1)(o)
            model = keras.models.Model(inputs=[i], outputs=[o])
            onnx_model = keras2onnx.convert_keras(model, model.name)
            data = np.random.rand(actual_batch_size, timesteps, input_dim).astype(np.float32).reshape((actual_batch_size, timesteps, input_dim))
            expected = model.predict(data)
            self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, self.model_files))

    def test_bert(self):
        from transformers import TFBertModel, BertTokenizer, TFBertForSequenceClassification
        '''
        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        '''
        import json
        raw_data = json.dumps({
            'text': 'My python is not working'
        })
        text = json.loads(raw_data)['text']
        # model = TFBertModel.from_pretrained(pretrained_weights)

        labels = ['c#', '.net', 'java', 'asp.net', 'c++', 'javascript', 'php', 'python', 'sql', 'sql-server']
        model_dir = 'bert-stackoverflow-v2'  # os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'bert-stackoverflow-v2')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # model = TFBertForSequenceClassification.from_pretrained(model_dir, num_labels=len(labels))
        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))
        inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='tf')
        inputs_onnx = {}
        for input_ in inputs:
            inputs_onnx[input_] = inputs[input_].numpy()

        # predictions = model.predict(inputs, steps=1)
        predictions = model.predict(inputs)
        print(predictions[0])
        # model.save_weights('bertv2.h5')
        onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=10)
        import onnx
        onnx.save_model(onnx_model, 'bertv2.onnx')
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))



if __name__ == "__main__":
    unittest.main()
