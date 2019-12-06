###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import functools
import onnxconverter_common
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnxconverter_common.onnx_ops import *  # noqa:

from .interim import OnnxObjectContainer, InterimContext
from ..proto import onnx_proto


class OnnxOperatorBuilder:
    def __init__(self, container, scope):
        # type: (OnnxOperatorBuilder, OnnxObjectContainer, InterimContext) -> None
        self._container = container
        self._scope = scope
        self.int32 = onnx_proto.TensorProto.INT32
        self.int64 = onnx_proto.TensorProto.INT64
        self.float = onnx_proto.TensorProto.FLOAT
        self.double = onnx_proto.TensorProto.DOUBLE
        self.bool = onnx_proto.TensorProto.BOOL

        apply_operations = onnxconverter_common.onnx_ops.__dict__
        for k_, m_ in apply_operations.items():
            if k_.startswith("apply_") and callable(m_):
                setattr(self, k_, functools.partial(self.apply_op, m_))

    def _process_inputs(self, inputs, name):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        ox_inputs = []
        for i_ in inputs:
            ox_n = i_
            if isinstance(i_, np.ndarray):
                ox_n = self._scope.get_unique_variable_name(name + '_i')
                self._container.add_initializer(
                    ox_n,
                    NP_TYPE_TO_TENSOR_TYPE[i_.dtype],
                    i_.shape,
                    i_.flatten()
                )
            elif isinstance(i_, (tuple, list)):
                ox_n = self._scope.get_unique_variable_name(name + i_[0])
                self._container.add_initializer(
                    ox_n,
                    i_[1],
                    i_[2].shape,
                    i_[2].flatten()
                )
            elif isinstance(i_, str):
                pass
            else:
                raise RuntimeError('Unknown type for ONNX initializer: {}'.format(type(i_)))
            ox_inputs.append(ox_n)

        return ox_inputs

    def add_node_all(self, op_type, inputs, name, outputs_num=1, op_domain='', op_version=None, **attrs):
        if op_version is None:
            op_version = self._container.target_opset
        outputs = [self._scope.get_unique_variable_name(name + str(i_)) for i_ in range(outputs_num)]
        ox_inputs = self._process_inputs(inputs, name)
        self._container.add_node(op_type, ox_inputs, outputs, op_domain, op_version, name=name, **attrs)
        return outputs if outputs_num > 1 else outputs[0]

    def add_node(self, op_type, inputs, name, op_domain='', op_version=None, **attrs):
        return self.add_node_all(op_type, inputs, name, 1, op_domain, op_version, **attrs)

    def add_node_with_output(self, op_type, inputs, outputs, name, op_domain='', op_version=None, **attrs):
        if op_version is None:
            op_version = self._container.target_opset
        ox_inputs = self._process_inputs(inputs, name)
        self._container.add_node(op_type, ox_inputs, outputs, op_domain, op_version, name=name, **attrs)
        return outputs

    def apply_op(self, apply_func, inputs, name, outputs_num=1, **attrs):
        outputs = [self._scope.get_unique_variable_name(name + str(i_)) for i_ in range(outputs_num)]
        ox_inputs = self._process_inputs(inputs, name)
        apply_func(self._scope, ox_inputs, outputs, self._container, operator_name=name, **attrs)
        return outputs if outputs_num > 1 else outputs[0]

    def apply_op_with_output(self, apply_func_name, inputs, outputs, name, **attrs):
        apply_operations = onnxconverter_common.onnx_ops.__dict__
        apply_func = apply_operations[apply_func_name]
        assert apply_func is not None, "{} not implemented in onnx_ops.py.".format(apply_func_name)
        ox_inputs = self._process_inputs(inputs, name)
        apply_func(self._scope, ox_inputs, outputs, self._container, operator_name=name, **attrs)

