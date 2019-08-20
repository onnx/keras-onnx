import os
import fire
import onnx
import tensorflow as tf
from .main import convert_tensorflow, convert_keras


def main(input_file, output_file=None, inputs=None, outputs=None, opset=None, channel_first=None):
    """
    A command line interface for Keras/Tensorflow model to ONNX converter.
    :param input_file: the original model file path
    :param output_file: the converted ONNX model file path (optional)
    :param inputs: The model graph input node list (tensorflow model only)
    :param outputs: The model graph output node list (tensorflow model only)
    :param opset: the target opset for the ONNX model.
    :param channel_first: the input name needs to be transposed as NCHW
    :return:
    """

    file_ext = os.path.splitext(input_file)
    if output_file is None:
        output_file = file_ext[0] + '.onnx'

    v_channel_first = [] if channel_first is None else channel_first.split(sep=',')
    v_inputs = [] if inputs is None else inputs.split(sep=',')
    if file_ext[-1] == '.h5':
        kml = tf.keras.models.load_model(input_file)
        oxml = convert_keras(kml, kml.model, '', opset, channel_first)

    else:
        with tf.gfile.GFile(input_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            oxml = convert_tensorflow(graph_def,
                                      os.path.basename(input_file),
                                      v_inputs,
                                      outputs.split(sep=','),
                                      '',
                                      opset,
                                      v_channel_first)

    onnx.save_model(oxml, output_file)


if __name__ == '__main__':
    # the color output doesn't work on some Windows cmdline tools
    if os.environ.get('OS', '') == 'Windows_NT':
        os.environ.update(ANSI_COLORS_DISABLED='1')
    fire.Fire(main)
