###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from onnxconverter_common.onnx_ex import make_model_ex
from .common import k2o_logger
from .common import OnnxObjectContainer, Variable, InterimContext
from .common.data_types import TensorType, Int64Type, FloatType, StringType
from .funcbook import get_converter
from .proto import helper


class KerasTfModelContainer(object):
    def __init__(self, model, graph):
        self._input_raw_names = list()
        self._output_raw_names = list()
        self.tf_graph = graph
        self.model = model

    @property
    def raw_model(self):
        return self.tf_graph

    def add_input_name(self, name):
        # The order of adding strings matters. The final model's input names are sequentially added as this list
        if name not in self._input_raw_names:
            self._input_raw_names.append(name)

    def add_output_name(self, name):
        # The order of adding strings matters. The final model's output names are sequentially added as this list
        if name not in self._output_raw_names:
            self._output_raw_names.append(name)

    @property
    def input_names(self):
        return [name for name in self._input_raw_names]

    @property
    def output_names(self):
        return [name for name in self._output_raw_names]


class Topology:

    def __init__(self, model, graph, target_opset=None, custom_op_dict=None,
                 reserved_variable_names=None, reserved_operator_names=None):
        """
        Initialize a Topology object, which is an intermediate representation of a computational graph.

        :param model: RawModelContainer object or one of its derived classes. It contains the original model.
        :param default_batch_size: batch_size prepend to scalar and array types
        :param initial_types: A list providing some types for some root variables. Each element is a tuple of a variable
        name and a type defined in data_types.py.
        :param target_opset: the onnx model targeted opset number.
        :param reserved_variable_names: A set of strings which are not allowed to be used as a variable name
        :param reserved_operator_names: A set of strings which are not allowed to be used as a operator name
        """
        self.scopes = []
        self.raw_model = KerasTfModelContainer(model, graph)
        self.scope_names = set()
        self.variable_name_set = reserved_variable_names if reserved_variable_names is not None else set()
        self.operator_name_set = reserved_operator_names if reserved_operator_names is not None else set()
        self.target_opset = target_opset
        self.debug_mode = False
        self.custom_op_dict = {} if custom_op_dict is None else custom_op_dict

        # This attribute is used in optimizing the graph structure. If root_names is not empty, only the variables
        # specified will be treated as the roots (i.e., set is_fed to True in the beginning of a graph evaluation) of
        # the graph. Specifying all root variables in this list and leaving it empty are equivalent. This attribute
        # directly affects initialize_graph_status_for_traversing function and indirectly affects _infer_all_shapes and
        # _prune functions.
        self.root_names = list()

    def get_unique_scope_name(self, seed):
        return Variable.generate_unique_name(seed, self.scope_names)

    def declare_scope(self, seed, parent_scopes=None):
        scope = InterimContext(self.get_unique_scope_name(seed), parent_scopes, self.variable_name_set,
                               self.operator_name_set, self.target_opset)
        self.scopes.append(scope)
        return scope

    def unordered_operator_iterator(self):
        for scope in self.scopes:
            for operator in scope.operators.values():
                yield operator

    def unordered_variable_iterator(self):
        for scope in self.scopes:
            for variable in scope.variables.values():
                yield variable

    def topological_operator_iterator(self):
        """
        This is an iterator of all operators in Topology object. Operators may be produced in a topological order.
        If you want to simply go though all operators without considering their topological structure, please use
        another function, unordered_operator_iterator.
        """
        self.initialize_graph_status_for_traversing()
        while not all(operator.is_evaluated for scope in self.scopes for operator in scope.operators.values()):
            is_evaluation_happened = False
            for operator in self.unordered_operator_iterator():
                if not operator.is_evaluated:
                    # Make this operator as handled
                    operator.is_evaluated = True
                    is_evaluation_happened = True
                    # Send out an operator
                    yield operator
            # After scanning through the whole computational graph, at least one operator should be evaluated. If not,
            # we need to terminate this procedure to avoid dead lock.
            if not is_evaluation_happened:
                break

    def _check_structure(self):
        """
        This function applies some rules to check if the parsed model is proper. Currently, it only checks if isolated
        variable and isolated operator exists.
        """
        # Collect all variable names and operator names
        unused_variables = set()
        unused_operators = set()
        for variable in self.unordered_variable_iterator():
            unused_variables.add(variable.full_name)
        for operator in self.unordered_operator_iterator():
            unused_operators.add(operator.full_name)

        for operator in self.unordered_operator_iterator():
            for variable in operator.inputs:
                # A variable is used by an operator, so we remove the variable from the unused-variable list.
                unused_variables.discard(variable.full_name)
                # A operator has an input, so we remove the operator from the unused-operator list.
                unused_operators.discard(operator.full_name)
            for variable in operator.outputs:
                # A variable is used by an operator, so we remove the variable from the unused-variable list.
                unused_variables.discard(variable.full_name)
                # A operator has an output, so we remove the operator from the unused-operator list.
                unused_operators.discard(operator.full_name)
            for variable in operator.input_masks + operator.output_masks:
                if variable is None: continue
                # A variable is used by an operator, so we remove the variable from the unused-variable list.
                unused_variables.discard(variable.full_name)
                # A operator has an output, so we remove the operator from the unused-operator list.
                unused_operators.discard(operator.full_name)

        if len(unused_variables) > 0:
            raise RuntimeError('Isolated variables exist: %s' % unused_variables)

        # if len(unused_operators) > 0:
        #     raise RuntimeError('Isolated operators exist: %s' % unused_operators)

    def initialize_graph_status_for_traversing(self):
        """
        Initialize the status of all variables and operators for traversing the underline graph
        """
        # In the beginning, we set is_root and is_leaf true. For is_fed, we have two different behaviors depending on
        # whether root_names is empty.
        for variable in self.unordered_variable_iterator():
            # If root_names is set, we only set those variable to be fed. Otherwise, all roots would be fed.
            if self.root_names:
                if variable.onnx_name in self.root_names:
                    variable.is_fed = True
                else:
                    variable.is_fed = False
            else:
                variable.is_fed = True
            variable.is_root = True
            variable.is_leaf = True

        # Then, we flip some flags by applying some simple rules so that only
        #   1. all roots get is_root=True and is_fed=True
        #   2. all leaves get is_leaf=True
        for operator in self.unordered_operator_iterator():
            operator.is_evaluated = False  # All operators are not processed in the beginning
            for variable in operator.outputs:
                # Output cannot be fed before graph traversing
                variable.is_fed = False
                # If the variable is an output of one operator, it must not be a root
                variable.is_root = False
            for variable in operator.inputs:
                # If the variable is an input of one operator, it must not be a leaf
                variable.is_leaf = False

    def compile(self):
        """
        This function aims at giving every operator enough information so that all operator conversions can happen
        independently. We also want to check, fix, and simplify the network structure here.
        """
        self._check_structure()


def _remove_unused_initializers(nodes, initializers):
    adjusted_initializers = []
    nodes_input_set = set()
    for n_ in nodes:
        for input_name_ in n_.input:
            nodes_input_set.add(input_name_)

    for initializers_ in initializers:
        if initializers_.name in nodes_input_set:
            adjusted_initializers.append(initializers_)

    return adjusted_initializers


def _remove_unused_nodes(nodes, inputs, outputs):
    nodes_input_set = set()
    for n_ in nodes:
        for input_name_ in n_.input:
            nodes_input_set.add(input_name_)

    input_dict = set([in_.name for in_ in inputs])
    output_dict = {}
    for nd_ in nodes:
        output_dict.update({o_: nd_ for o_ in nd_.output})

    nodes_to_keep = set()
    node_inputs = [output_dict[ts_.name] for ts_ in outputs]
    while node_inputs:
        nd_ = node_inputs[0]
        del node_inputs[0]
        if id(nd_) in nodes_to_keep:
            continue

        nodes_to_keep.add(id(nd_))
        for in_ in nd_.input:
            if in_ in output_dict:
                node_inputs.append(output_dict[in_])
            else:
                assert in_ in input_dict

    return [nd_ for nd_ in nodes if id(nd_) in nodes_to_keep]


def _build_extra_inputs(container):
    # When calling ModelComponentContainer's add_initializer(...), nothing is added into the input list.
    # However, In ONNX, for target opset < 9, initializers should also be model's (GraphProto) inputs.
    # Thus, we create ValueInfoProto objects from initializers (type: TensorProto) directly and then add them into model's input list.
    extra_inputs = []  # ValueInfoProto list of the initializers
    for tensor in container.initializers:
        # Sometimes (especially when creating optional input values such as RNN's initial hidden state), an initializer
        # is also one of the original model's input, so it has been added into the container's input list. If this is
        # the case, we need to skip one iteration to avoid duplicated inputs.
        if tensor.name in [value_info.name for value_info in container.inputs]:
            continue

        # Initializers are always tensors so we can just call make_tensor_value_info(...)
        value_info = helper.make_tensor_value_info(tensor.name, tensor.data_type, tensor.dims)
        extra_inputs.append(value_info)
    return extra_inputs


def convert_topology(topology, model_name, doc_string, target_opset, channel_first_inputs=None):
    """
    This function is used to convert our Topology object defined in _parser.py into a ONNX model (type: ModelProto).
    :param topology: The Topology object we are going to convert
    :param model_name: GraphProto's name. Let "model" denote the returned model. The string "model_name" would be
    assigned to "model.graph.name."
    :param doc_string: A string attached to the produced model
    :param target_opset: The maximun opset number in the model.
    :param channel_first_inputs: A list of channel first input.
    :return: a ONNX ModelProto
    """
    topology.initialize_graph_status_for_traversing()

    container = OnnxObjectContainer(target_opset)

    # Put roots and leaves as ONNX's model into buffers. They will be added into ModelComponentContainer later.
    tensor_inputs = {}
    other_inputs = {}
    tensor_outputs = {}
    other_outputs = {}
    for scope in topology.scopes:
        for variable in scope.variables.values():
            if variable.is_root:
                if isinstance(variable.type, (TensorType, Int64Type, FloatType, StringType)):
                    tensor_inputs[variable.raw_name] = variable
                else:
                    other_inputs[variable.raw_name] = variable
            if variable.is_leaf:
                if isinstance(variable.type, (TensorType, Int64Type, FloatType, StringType)):
                    tensor_outputs[variable.raw_name] = variable
                else:
                    other_outputs[variable.raw_name] = variable

    # Add roots the graph according to their order in the original model
    nchw_inputs = []
    if channel_first_inputs is None:
        channel_first_inputs = []
    for name in topology.raw_model.input_names:
        if name in tensor_inputs:
            onnx_input = tensor_inputs[name]  # type: Variable
            if name in channel_first_inputs or \
                    (name.endswith(':0') and name[:-2] in channel_first_inputs):
                nchw_inputs.append(onnx_input.full_name)
                s = onnx_input.type.shape
                onnx_input.type.shape = [s[0], s[3], s[1], s[2]]
            container.add_input(onnx_input)

    for name in topology.raw_model.input_names:
        if name in other_inputs:
            container.add_input(other_inputs[name])

    # Add leaves the graph according to their order in the original model
    for name in topology.raw_model.output_names:
        if name in tensor_outputs:
            container.add_output(tensor_outputs[name])
    for name in topology.raw_model.output_names:
        if name in other_outputs:
            container.add_output(other_outputs[name])

    # Traverse the graph from roots to leaves
    for operator in topology.topological_operator_iterator():
        scope = next(scope for scope in topology.scopes if scope.name == operator.scope)
        k2o_logger().debug("Converting the operator (%s): %s" % (operator.full_name, operator.type))
        cvt = get_converter(operator.type)
        if cvt is None:
            raise RuntimeError("Unexpected error on find the converter for op {}".format(operator.type))
        cvt(scope, operator, container)

    # enable the ONNX optimizations
    graph = None
    extra_inputs = _build_extra_inputs(container)
    nodes = _remove_unused_nodes(container.nodes, container.inputs + extra_inputs, container.outputs)

    if not topology.debug_mode:
        try:
            import onnxconverter_common
            origin_node_number = len(nodes)
            if target_opset < 9:
                nodes = onnxconverter_common.optimizer.optimize_onnx(nodes, nchw_inputs=nchw_inputs,
                                                                     inputs=container.inputs + extra_inputs,
                                                                     outputs=container.outputs)
                node_number = len(nodes)
            else:
                graph = onnxconverter_common.optimizer.optimize_onnx_graph(nodes, nchw_inputs=nchw_inputs,
                                                                           inputs=container.inputs,
                                                                           outputs=container.outputs,
                                                                           initializers=container.initializers,
                                                                           model_value_info=container.value_info,
                                                                           model_name=model_name,
                                                                           target_opset=container.target_opset)
                node_number = len(graph.node)
            k2o_logger().info(
                "The ONNX operator number change on the optimization: {} -> {}".format(origin_node_number, node_number))
        except ImportError:
            onnx_not_imported = 'onnxconverter_common is not imported,'
            if nchw_inputs:
                raise Exception(
                    '{} nchw_inputs does not make effect. Please set nchw_inputs to empty.'.format(onnx_not_imported))
            k2o_logger().warning('{} so the convertor optimizer is not enabled.'.format(onnx_not_imported))
        except Exception as e:  # noqa
            # either optimizer issue or converter issue, we just let it go to diagnose the issue from the converted model.
            k2o_logger().warning(
                'There is an error({}) happened during optimizing on the converted model!'.format(type(e)))
            k2o_logger().warning(str(e))
            import traceback
            tb = traceback.format_exc()
            k2o_logger().warning(tb)

    if graph is None:
        # Create a graph from its main components
        adjusted_initializers = _remove_unused_initializers(nodes, container.initializers)
        if target_opset < 9:
            adjusted_extra_inputs = _remove_unused_initializers(nodes, extra_inputs)
            graph = helper.make_graph(nodes, model_name, container.inputs + adjusted_extra_inputs,
                                      container.outputs, adjusted_initializers)
        else:
            graph = helper.make_graph(nodes, model_name, container.inputs,
                                      container.outputs, adjusted_initializers)

        # Add extra information related to the graph
        graph.value_info.extend(container.value_info)

    # Create model
    onnx_model = make_model_ex(graph,
                               container.node_domain_version_pair_sets,
                               target_opset, doc_string=doc_string)
    return onnx_model
