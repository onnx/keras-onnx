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


class TestSegNet(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def test_segnet(self):
        # From https://github.com/divamgupta/image-segmentation-keras/models/segnet.py
        model = keras_segmentation.models.segnet.segnet(101)
        test_image(model, target_size=(416, 608))

    def test_vgg_segnet(self):
        # From https://github.com/divamgupta/image-segmentation-keras/models/segnet.py
        model = keras_segmentation.models.segnet.vgg_segnet(101)
        test_image(model, target_size=(416, 608))


if __name__ == "__main__":
    unittest.main()
