# Introduction 
The original keras model was coming from: <https://github.com/qqwweee/keras-yolo3>, clone the project and follow the 'Quick Start' to get the pre-trained model.

# Convert
```
export PYTHONPATH=$(the keras-yolo3 path) 
python yolov3.py -c  # convert the model into the ONNX
python yolov3.py data/test.jpg # run a detect example
```
