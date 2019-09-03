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
from test_utils import run_image
img_path = os.path.join(os.path.dirname(__file__), '../data', 'street.jpg')


class TestFCN(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def test_unet(self):
        # From https://github.com/divamgupta/image-segmentation-keras/models/unet.py
        model = keras_segmentation.models.unet.unet(101)
        res = run_image(model, self.model_files, img_path, target_size=(416, 608))
        self.assertTrue(*res)


if __name__ == "__main__":
    unittest.main()
