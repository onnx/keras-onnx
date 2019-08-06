###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################


class Operator:
    """
    The intermediate object to store the information for the final ONNX operator generation.
    """
    def __init__(self, onnx_name, scope, type, raw_operator, target_opset):
        """
        :param onnx_name: A unique ID, which is a string
        :param scope: The name of the scope where this operator is declared. It's a string.
        :param type: A object which uniquely characterizes the type of this operator.
        :param raw_operator: The original operator which defines this operator;
        :param target_opset: the target opset.
        """
        self.onnx_name = onnx_name  # operator name in the converted model
        self.scope = scope
        self.type = type
        self.raw_operator = raw_operator
        self.inputs = []
        self.input_mask = None
        self.outputs = []
        self.output_mask = None
        self.nodelist = None
        self.is_evaluated = None
        self.target_opset = target_opset
        self.shape_infer = None
        self.tf2onnx_graph = None
        self.attrs = {}

    @property
    def full_name(self):
        """
        Return a globally unique operator ID
        """
        return self.onnx_name

    @property
    def input_full_names(self):
        """
        Return all input variables' names
        """
        return [variable.full_name for variable in self.inputs]

    @property
    def output_full_names(self):
        """
        Return all output variables' names
        """
        return [variable.full_name for variable in self.outputs]

    @property
    def original_operator(self):
        """
        Return the original operator/layer
        """
        return self.raw_operator

    def add_input(self, var):
        if self not in var.op_to:
            var.op_to.append(self)
        self.inputs.append(var)

    def add_output(self, var):
        assert var.op_from is None
        var.op_from = self
        self.outputs.append(var)

    def add_input_mask(self, var):
        if self not in var.op_to:
            var.op_to.append(self)
        self.input_mask = var

    def add_output_mask(self, var):
        if var.op_from is None:
            var.op_from = self
        self.output_mask = var

    def update_attrs(self, **attrs):
        self.attrs.update(attrs)

    def get_attr(self, key):
        return self.attrs.get(key, None)

    def get_input_shapes(self):
        return self.attrs.get('input_shapes', None)

    def get_output_shape(self):
        output_shape = self.outputs[0].type.shape
        if output_shape is None:
            output_shape = self.raw_operator.output_shape
        return output_shape
