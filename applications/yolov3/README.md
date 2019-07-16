# Introduction 
The original keras model was coming from: <https://github.com/qqwweee/keras-yolo3>, clone the project and follow the 'Quick Start' to get the pre-trained model.

We have converted yolov3 model successfully and uploaded to the model zoo <https://github.com/onnx/models/tree/master/yolov3>

# Convert
```
export PYTHONPATH=$(the keras-yolo3 path)
# run object detection, convert the model to onnx first if the onnx model does not exist
python yolov3.py <image url>
```