We have verified the following pre-trained keras models can be converted to onnx successfully.

Computer Vision

|  Model Name        | Extra Instructions(*) |
|----------|-------|
| Xception | |
| VGG16, VGG19 | |
| ResNet50 | |
| InceptionV3, InceptionResNetV2 | |
| MobileNet, MobileNetV2 | |
| DenseNet121, DenseNet169, DenseNet201 | |
| NASNetMobile, NASNetLarge | |
| [YOLOv3](https://github.com/qqwweee/keras-yolo3) | [Readme](https://github.com/onnx/keras-onnx/tree/master/applications/yolov3)|
| [Mask RCNN](https://github.com/matterport/Mask_RCNN) | [Readme](https://github.com/onnx/keras-onnx/tree/master/applications/mask_rcnn)|

By default, the conversion is straightforward (see [Readme](https://github.com/onnx/keras-onnx/blob/master/README.md)). (*) shows the extra instructions that is needed.    

GAN

|  Model Name        | Extra Instructions(*) |
|----------|-------|
| [ACGAN (Auxiliary Classifier GAN)](https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py) ||
