###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import os
import sys
import unittest
import keras_segmentation
from os.path import dirname, abspath

sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../tests/'))
from test_utils import test_image


class TestFCN(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def test_unet(self):
        # From https://github.com/divamgupta/image-segmentation-keras/models/unet.py
        model = keras_segmentation.models.unet.unet(101)
        self._test_keras_model(model, target_size=(416, 608))


if __name__ == "__main__":
    unittest.main()
