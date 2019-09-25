# keras2onnx

|          | Linux | Windows |
|----------|-------|---------|
| keras.io | [![Build Status](https://dev.azure.com/onnxmltools/ketone/_apis/build/status/linux-conda-ci?branchName=master)](https://dev.azure.com/onnxmltools/ketone/_build/latest?definitionId=9&branchName=master) | [![Build Status](https://dev.azure.com/onnxmltools/ketone/_apis/build/status/win32-conda-ci?branchName=master)](https://dev.azure.com/onnxmltools/ketone/_build/latest?definitionId=10&branchName=master) | 
| tf.keras | [![Build Status](https://dev.azure.com/onnxmltools/ketone/_apis/build/status/linux-tf-keras-ci?branchName=master)](https://dev.azure.com/onnxmltools/ketone/_build/latest?definitionId=19&branchName=master) | [![Build Status](https://dev.azure.com/onnxmltools/ketone/_apis/build/status/win32-tf-keras-CI?branchName=master)](https://dev.azure.com/onnxmltools/ketone/_build/latest?definitionId=20&branchName=master) | 


# Introduction
The keras2onnx model converter enables users to convert Keras models into the [ONNX](https://onnx.ai) model format.
Initially, the Keras converter was developed in the project [onnxmltools](https://github.com/onnx/onnxmltools). keras2onnx converter development was moved into an [independent repository](https://github.com/onnx/keras-onnx) to support more kinds of Keras models and reduce the complexity of mixing multiple converters.

All Keras layers have been supported for conversion using keras2onnx since **ONNX opset 7**. Please refer to the [Keras documentation](https://keras.io/layers/about-keras-layers/) for details on Keras layers. The keras2onnx converter also supports the lambda/custom layer by working with the [tf2onnx](https://github.com/onnx/tensorflow-onnx) converter which is embedded directly into the source tree to avoid version conflicts and installation complexity.

Windows Machine Learning (WinML) users can use [WinMLTools](https://docs.microsoft.com/en-us/windows/ai/windows-ml/convert-model-winmltools) to convert their Keras models to the ONNX format. If you want to use the keras2onnx converter, please refer to the [WinML Release Notes](https://docs.microsoft.com/en-us/windows/ai/windows-ml/release-notes) to identify the corresponding ONNX opset for your WinML version.

keras2onnx has been tested on **Python 3.5, 3.6, and 3.7** (CI build). It does not support **Python 2.x**.

# Notes

# tf.keras v.s. keras.io
Both Keras model types are now supported in the keras2onnx converter. If the user's Keras package was installed from [Keras.io](https://keras.io/), the converter converts the model as it was created by the keras.io package. Otherwise, it will convert it through [tf.keras](https://www.tensorflow.org/guide/keras).<br>

If you want to override this behaviour, please specify the environment variable TF_KERAS=1 before invoking the converter python API.

# Usage
Before running the converter, please notice that tensorflow has to be installed in your python environment,
you can choose **tensorflow** package(CPU version) or **tensorflow-gpu**(GPU version)

# Validated pre-trained Keras models
We converted successfully for all the keras application models, and several other pretrained models. See below:

|  Model Name        | Category | Notes |
|----------|-------|------|
| Xception | Computer Vision |
| VGG16, VGG19 | Computer Vision |
| ResNet50 | Computer Vision |
| InceptionV3, InceptionResNetV2 | Computer Vision |
| MobileNet, MobileNetV2 | Computer Vision |
| DenseNet121, DenseNet169, DenseNet201 | Computer Vision |
| NASNetMobile, NASNetLarge | Computer Vision |
| [FCN, FCN-Resnet](https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/fcn.py) | Computer Vision |
| [PSPNet](https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/pspnet.py) | Computer Vision |
| [Segnet, VGG-Segnet](https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/segnet.py) | Computer Vision |
| [UNet](https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/unet.py) | Computer Vision |
| [LPCNet](https://github.com/mozilla/LPCNet) | Speech |
| [ACGAN (Auxiliary Classifier GAN)](https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py) | GAN |
| [Adversarial Autoencoder](https://github.com/eriklindernoren/Keras-GAN/blob/master/aae/aae.py) | GAN |
| [BGAN (Boundary-Seeking GAN)](https://github.com/eriklindernoren/Keras-GAN/blob/master/bgan/bgan.py) | GAN |
| [BIGAN (Bidirectional GAN)](https://github.com/eriklindernoren/Keras-GAN/blob/master/bigan/bigan.py) | GAN |
| [CGAN (Conditional GAN)](https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py) | GAN |
| [Coupled GAN](https://github.com/eriklindernoren/Keras-GAN/blob/master/cogan/cogan.py) | GAN |
| [Deep Convolutional GAN](https://github.com/eriklindernoren/Keras-GAN/blob/master/discogan/discogan.py) | GAN |
| [DualGAN](https://github.com/eriklindernoren/Keras-GAN/blob/master/dualgan/dualgan.py) | GAN |
| [Generative Adversarial Network](https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py) | GAN |
| [InfoGAN](https://github.com/eriklindernoren/Keras-GAN/blob/master/infogan/infogan.py) | GAN |
| [LSGAN](https://github.com/eriklindernoren/Keras-GAN/blob/master/lsgan/lsgan.py) | GAN |
| [Pix2Pix](https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py) | GAN |
| [Semi-Supervised GAN](https://github.com/eriklindernoren/Keras-GAN/blob/master/sgan/sgan.py) | GAN |
| [Super-Resolution GAN](https://github.com/eriklindernoren/Keras-GAN/blob/master/srgan/srgan.py) | GAN |
| [Wasserstein GAN](https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py) | GAN |
| [Wasserstein GAN GP](https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py) | GAN |
| [keras-team examples](https://github.com/keras-team/keras/blob/master/examples/) | Text and Sequence | addition_rnn, babi_rnn, imdb_bidirectional_lstm, imdb_cnn_lstm, imdb_lstm, lstm_text_generation, reuters_mlp |

The following models need customed conversion, see the instruction column.

|  Model Name        | Category | Instruction |
|----------|-------|-------|
| [YOLOv3](https://github.com/qqwweee/keras-yolo3) | Computer Vision | [Readme](https://github.com/onnx/keras-onnx/tree/master/applications/yolov3)|
| [Mask RCNN](https://github.com/matterport/Mask_RCNN) | Computer Vision | [Readme](https://github.com/onnx/keras-onnx/tree/master/applications/mask_rcnn)|
| [Context-Conditional GAN](https://github.com/eriklindernoren/Keras-GAN/tree/master/ccgan) | GAN | [Unit test](https://github.com/onnx/keras-onnx/blob/master/applications/nightly_build/test_ccgan.py)|


## Scripts
It will be useful to convert the models from Keras to ONNX from a python script.
You can use the following API:
```
import keras2onnx
keras2onnx.convert_keras(model, name=None, doc_string='', target_opset=None, channel_first_inputs=None):
    # type: (keras.Model, str, str, int, []) -> onnx.ModelProto
    """
    :param model: keras model
    :param name: the converted onnx model internal name
    :param doc_string:
    :param target_opset:
    :param channel_first_inputs: A list of channel first input.
    :return:
    """
```

Use the following script to convert keras application models to onnx, and then perform inference:
```
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import keras2onnx
import onnxruntime

# image preprocessing
img_path = 'street.jpg'   # make sure the image is in img_path
img_size = 224
img = image.load_img(img_path, target_size=(img_size, img_size))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# load keras model
from keras.applications.resnet50 import ResNet50
model = ResNet50(include_top=True, weights='imagenet')

# convert to onnx model
onnx_model = keras2onnx.convert_keras(model, model.name)

# runtime prediction
content = onnx_model.SerializeToString()
sess = onnxruntime.InferenceSession(content)
x = x if isinstance(x, list) else [x]
feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
pred_onnx = sess.run(None, feed)
```

An alternative way to load onnx model to runtime session is to save the model first:
```
import onnx
temp_model_file = 'model.onnx'
onnx.save_model(onnx_model, temp_model_file)
sess = onnxruntime.InferenceSession(temp_model_file)
```

## Contribute
We welcome contributions in the form of feedback, ideas, or code.

## License
[MIT License](LICENSE)
