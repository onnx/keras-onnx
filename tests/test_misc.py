import unittest
import tensorflow as tf
import numpy as np

from keras2onnx.subgraph import create_subgraph


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
                tf.placeholder(dtype=np.float32),
                name="add")

        self.assertNotEqual(t_add.op.inputs[0].op.type, 'Placeholder')
        node_list = g.get_operations()
        node_list.remove(i0.op)
        sgv, replacement = create_subgraph(node_list)
        self.assertEqual(sgv.graph.get_operation_by_name(t_add.op.name).inputs[0].op.type, 'Placeholder')


if __name__ == '__main__':
    unittest.main()
