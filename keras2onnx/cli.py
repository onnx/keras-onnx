# SPDX-License-Identifier: Apache-2.0

import os
import fire
import onnx
import tensorflow as tf
from .main import convert_keras


def main(input_file, output_file=None, opset=None, channel_first=None):
    """
    A command line interface for Keras model to ONNX converter.
    :param input_file: the original model file path, could be a folder name of TF saved model
    :param output_file: the converted ONNX model file path (optional)
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

    assert file_ext[-1] == '.h5', "Unknown file extension."
    kml = tf.keras.models.load_model(input_file)
    oxml = convert_keras(kml, kml.name, '', opset, channel_first)
    onnx.save_model(oxml, output_file)


if __name__ == '__main__':
    # the color output doesn't work on some Windows cmdline tools
    if os.environ.get('OS', '') == 'Windows_NT':
        os.environ.update(ANSI_COLORS_DISABLED='1')
    fire.Fire(main)
