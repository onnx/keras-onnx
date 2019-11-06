# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for while loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf

from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test


class LoopTests(Tf2OnnxBackendTestBase):

    def test_fill(self):
        from keras import backend as K
        class_box_scores = K.variable(np.random.random((2, 3)))
        c = 20
        classes_ = K.ones_like(class_box_scores, 'int32') * K.constant(value=c, dtype='int32')
        _ = tf.identity(classes_, name="output")
        input_names_with_port = []
        feed_dict = {}
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-06)


if __name__ == '__main__':
    unittest_main()
