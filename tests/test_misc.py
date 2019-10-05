import unittest
import tensorflow as tf
import numpy as np

from keras2onnx.subgraph import create_subgraph
import keras2onnx.common as _cmn
import keras2onnx.proto as _proto


class SubgraphTestCase(unittest.TestCase):
    """
    Tests for subgraph operations
    """

    def test_main(self):
        g = tf.Graph()
        with g.as_default():
            i0 = tf.constant(1.0, shape=[2, 3], name="a")
            t_add = tf.add(
                i0,
                tf.compat.v1.placeholder(dtype=np.float32),
                name="add")

        self.assertNotEqual(t_add.op.inputs[0].op.type, 'Placeholder')
        node_list = g.get_operations()
        node_list.remove(i0.op)
        sgv, replacement = create_subgraph(g, node_list, tf.compat.v1.Session())
        self.assertEqual(sgv.get_operation_by_name(t_add.op.name).inputs[0].op.type, 'Placeholder')


class ONNXOPSTestCase(unittest.TestCase):
    """
    Tests for ONNX Operator Builder
    """

    def test_apply(self):
        oopb = _cmn.onnx_ops.OnnxOperatorBuilder(_cmn.OnnxObjectContainer(_proto.get_opset_number_from_onnx()),
                                                 _cmn.InterimContext('_curr'))
        value = oopb.apply_add((np.array([[1.0], [0.5]], dtype='float32'),
                                ('_i1', oopb.float, np.array([2.0], dtype='float32'))), 'add')
        self.assertTrue(value[0].startswith('add'))


if __name__ == '__main__':
    unittest.main()
