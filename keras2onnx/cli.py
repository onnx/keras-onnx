import os
import fire
import onnx
import tensorflow as tf
from .main import convert_tensorflow, convert_keras


def main(input_file, output_file=None, inputs=None, outputs=None, opset=None, channel_first=None):
    """
    A command line interface for Keras/Tensorflow model to ONNX converter.
    :param input_file: the original model file path, could be a folder name of TF saved model
    :param output_file: the converted ONNX model file path (optional)
    :param inputs: The model graph input node list (tensorflow model only)
    :param outputs: The model graph output node list (tensorflow model only)
    :param opset: the target opset for the ONNX model.
    :param channel_first: the input name needs to be transposed as NCHW
    :return:
    """

    if not os.path.exists(input_file):
        print("File or directory name '{}' is invalid!".format(input_file))
        return

    file_ext = os.path.splitext(input_file)
    if output_file is None:
        output_file = file_ext[0] + '.onnx'

    v_channel_first = [] if channel_first is None else channel_first.split(sep=',')
    v_inputs = [] if inputs is None else inputs.split(sep=',')
    v_outputs = [] if outputs is None else outputs.split(sep=',')
    if file_ext[-1] == '.h5':
        kml = tf.keras.models.load_model(input_file)
        oxml = convert_keras(kml, kml.model, '', opset, channel_first)
    else:
        from tf2onnx import loader
        if os.path.isdir(input_file):
            chpt_file = [f_ for f_ in os.listdir(input_file) if f_.endswith('.meta')]
            if chpt_file:
                meta_file = os.path.join(input_file, chpt_file[0])
                graph_def, v_inputs, v_outputs = loader.from_checkpoint(meta_file, v_inputs, v_outputs)
            else:
                saved_pbfile = [f_ for f_ in os.listdir(input_file) if f_.endswith('.pb')]
                assert len(saved_pbfile) == 1
                saved_pbfile = os.path.join(input_file, saved_pbfile[0])
                graph_def, v_inputs, v_outputs = loader.from_saved_model(saved_pbfile, v_inputs, v_outputs)
        else:
            with tf.gfile.GFile(input_file, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
        oxml = convert_tensorflow(graph_def,
                                  os.path.basename(input_file),
                                  v_inputs,
                                  v_outputs,
                                  '',
                                  opset,
                                  v_channel_first)

    onnx.save_model(oxml, output_file)


if __name__ == '__main__':
    # the color output doesn't work on some Windows cmdline tools
    if os.environ.get('OS', '') == 'Windows_NT':
        os.environ.update(ANSI_COLORS_DISABLED='1')
    fire.Fire(main)
