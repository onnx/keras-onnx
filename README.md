<!--- SPDX-License-Identifier: Apache-2.0 -->

# keras2onnx

<b>
We stopped active development of keras2onnx and keras2onnx is now frozen to tf-2.3 and onnx-1.10.

To convert your Keras models you can head over to [tf2onnx](https://github.com/onnx/tensorflow-onnx) which can convert Tensorflow, Keras, Tflite and Tensorflow.js models. All keras2onnx unit tests have been added to the tf2onnx ci pipeline to make sure there are no avoidable regressions. The tf2onnx api [tf2onnx.convert.from_keras()](https://github.com/onnx/tensorflow-onnx#from_keras-tf-20-and-newer) is similar to the keras2onnx api and we hope transition is painless.

You can find a simple tutorial how to convert keras models using tf2onnx [here](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/keras-resnet50.ipynb).

If you run into issue or need help with the transition, please open issue against tf2onnx [here](https://github.com/onnx/tensorflow-onnx/issues).
</b>
<br/>
<br/>



|          | Linux | Windows |
|----------|-------|---------|
| keras.io | [![Build Status](https://dev.azure.com/onnxmltools/ketone/_apis/build/status/linux-conda-ci?branchName=master)](https://dev.azure.com/onnxmltools/ketone/_build/latest?definitionId=9&branchName=master) | [![Build Status](https://dev.azure.com/onnxmltools/ketone/_apis/build/status/win32-conda-ci?branchName=master)](https://dev.azure.com/onnxmltools/ketone/_build/latest?definitionId=10&branchName=master) |
| tf.keras | [![Build Status](https://dev.azure.com/onnxmltools/ketone/_apis/build/status/linux-tf-keras-ci?branchName=master)](https://dev.azure.com/onnxmltools/ketone/_build/latest?definitionId=19&branchName=master) | [![Build Status](https://dev.azure.com/onnxmltools/ketone/_apis/build/status/win32-tf-keras-CI?branchName=master)](https://dev.azure.com/onnxmltools/ketone/_build/latest?definitionId=20&branchName=master) |


# Introduction
The keras2onnx model converter enables users to convert Keras models into the [ONNX](https://onnx.ai) model format.
Initially, the Keras converter was developed in the project [onnxmltools](https://github.com/onnx/onnxmltools). keras2onnx converter development was moved into an [independent repository](https://github.com/onnx/keras-onnx) to support more kinds of Keras models and reduce the complexity of mixing multiple converters.

Most of the common Keras layers have been supported for conversion. Please refer to the [Keras documentation](https://keras.io/layers/about-keras-layers/) or [tf.keras docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers) for details on Keras layers.

Windows Machine Learning (WinML) users can use [WinMLTools](https://docs.microsoft.com/en-us/windows/ai/windows-ml/convert-model-winmltools) which wrap its call on keras2onnx to convert the Keras models. If you want to use the keras2onnx converter, please refer to the [WinML Release Notes](https://docs.microsoft.com/en-us/windows/ai/windows-ml/release-notes) to identify the corresponding ONNX opset number for your WinML version.

keras2onnx has been tested on **Python 3.5 - 3.8**, with **tensorflow 1.x/2.0 - 2.2**  (CI build). It does not support **Python 2.x**.

# Install
You can install latest release of Keras2ONNX from PyPi:

```
pip install keras2onnx
```
or install from source:

```
pip install -U git+https://github.com/microsoft/onnxconverter-common
pip install -U git+https://github.com/onnx/keras-onnx
```
Before running the converter, please notice that tensorflow has to be installed in your python environment,
you can choose **tensorflow**/**tensorflow-cpu** package(CPU version) or **tensorflow-gpu**(GPU version)

# Notes
Keras2ONNX supports the new Keras subclassing model which was introduced in tensorflow 2.0 since the version **1.6.5**. Some typical subclassing models like [huggingface/transformers](https://github.com/huggingface/transformers) have been converted into ONNX and validated by ONNXRuntime.<br>

Since its version 2.3, the [multi-backend Keras (keras.io)](https://keras.io/#multi-backend-keras-and-tfkeras) stops the support of the tensorflow version above 2.0. The auther suggests to switch to tf.keras for the new features.
## Multi-backend Keras and tf.keras:
Both Keras model types are now supported in the keras2onnx converter. If in the user python env, Keras package was installed from [Keras.io](https://keras.io/) and tensorflow package version is 1.x, the converter converts the model as it was created by the keras.io package. Otherwise, it will convert it through [tf.keras](https://www.tensorflow.org/guide/keras).<br>

If you want to override this behaviour, please specify the environment variable TF_KERAS=1 before invoking the converter python API.
# Development
Keras2ONNX depends on [onnxconverter-common](https://github.com/microsoft/onnxconverter-common). In practice, the latest code of this converter requires the latest version of onnxconverter-common, so if you install this converter from its source code, please install the onnxconverter-common in source code mode before keras2onnx installation.

# Validated pre-trained Keras models
Most Keras models could be converted successfully by calling ```keras2onnx.convert_keras```, including CV, GAN, NLP, Speech and etc. See the tutorial [here](https://github.com/onnx/keras-onnx/tree/master/tutorial). However some models with a lot of custom operations need custom conversion, the following are some examples,
like [YOLOv3](https://github.com/qqwweee/keras-yolo3), and [Mask RCNN](https://github.com/matterport/Mask_RCNN).


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

The inference result is a list which aligns with keras model prediction result `model.predict()`.
An alternative way to load onnx model to runtime session is to save the model first:
```
temp_model_file = 'model.onnx'
keras2onnx.save_model(onnx_model, temp_model_file)
sess = onnxruntime.InferenceSession(temp_model_file)
```

## Contribute
We welcome contributions in the form of feedback, ideas, or code.

## License
[Apache License v2.0](LICENSE)
