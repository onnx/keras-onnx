# SPDX-License-Identifier: Apache-2.0

# the interim objects for the model conversion.
import six
from ..proto import helper
from .intop import Operator


class OnnxObjectContainer(object):
    """
    In the conversion phase, this class is used to collect all materials required to build an ONNX GraphProto, which is
    encapsulated in a ONNX ModelProto.
    """

    def __init__(self, target_opset):
        '''
        :param targeted_onnx: A string, for example, '1.1.2' and '1.2'.
        '''
        # Inputs of ONNX graph. They are ValueInfoProto in ONNX.
        self.inputs = []
        # Outputs of ONNX graph. They are ValueInfoProto in ONNX.
        self.outputs = []
        # ONNX tensors (type: TensorProto). They are initializers of ONNX GraphProto.
        self.initializers = []
        # Intermediate variables in ONNX computational graph. They are ValueInfoProto in ONNX.
        self.value_info = []
        # ONNX nodes (type: NodeProto) used to define computation structure
        self.nodes = []
        # ONNX operators' domain-version pair set. They will be added into opset_import field in the final ONNX model.
        self.node_domain_version_pair_sets = set()
        # The targeted ONNX version. All produced operators should be supported by the targeted ONNX version.
        self.target_opset = target_opset
        # ONNX node name list
        self.node_names = {}

    @staticmethod
    def _make_value_info(variable):
        value_info = helper.ValueInfoProto()
        value_info.name = variable.full_name
        value_info.type.CopyFrom(variable.type.to_onnx_type())
        if variable.type.doc_string:
            value_info.doc_string = variable.type.doc_string
        return value_info

    def add_input(self, variable):
        '''
        Add our Variable object defined _parser.py into the the input list of the final ONNX model

        :param variable: The Variable object to be added
        '''
        self.inputs.append(self._make_value_info(variable))

    def add_output(self, variable):
        '''
        Add our Variable object defined _parser.py into the the output list of the final ONNX model

        :param variable: The Variable object to be added
        '''
        self.outputs.append(self._make_value_info(variable))

    def add_initializer(self, name, onnx_type, shape, content):
        '''
        Add a TensorProto into the initializer list of the final ONNX model

        :param name: Variable name in the produced ONNX model.
        :param onnx_type: Element types allowed in ONNX tensor, e.g., TensorProto.FLOAT and TensorProto.STRING.
        :param shape: Tensor shape, a list of integers.
        :param content: Flattened tensor values (i.e., a float list or a float array).
        '''
        if any(d is None for d in shape):
            raise ValueError('Shape of initializer cannot contain None')
        tensor = helper.make_tensor(name, onnx_type, shape, content)
        self.initializers.append(tensor)

    def add_initializer_by_name(self, scope, original_name, onnx_type, shape, content):
        if original_name not in scope.variable_name_mapping:
            onnx_name = scope.get_unique_variable_name(original_name)
            scope.variable_name_mapping[original_name] = [onnx_name]

            if any(d is None for d in shape):
                raise ValueError('Shape of initializer cannot contain None')
            tensor = helper.make_tensor(onnx_name, onnx_type, shape, content)
            self.initializers.append(tensor)
        else:
            onnx_name = scope.get_onnx_variable_name(original_name)
            assert next(ts_ for ts_ in self.initializers if ts_.name == onnx_name)
        return onnx_name

    def add_initializer_from_tensor(self, tensor):
        self.initializers.append(tensor)

    def add_value_info(self, variable):
        self.value_info.append(self._make_value_info(variable))

    def add_node(self, op_type, inputs, outputs, op_domain='', op_version=1, **attrs):
        '''
        Add a NodeProto into the node list of the final ONNX model. If the input operator's domain-version information
        cannot be found in our domain-version pool (a Python set), we may add it.

        :param op_type: A string (e.g., Pool and Conv) indicating the type of the NodeProto
        :param inputs: A list of strings. They are the input variables' names of the considered NodeProto
        :param outputs: A list of strings. They are the output variables' names of the considered NodeProto
        :param op_domain: The domain name (e.g., ai.onnx.ml) of the operator we are trying to add.
        :param op_version: The version number (e.g., 0 and 1) of the operator we are trying to add.
        :param attrs: A Python dictionary. Keys and values are attributes' names and attributes' values, respectively.
        '''

        if isinstance(inputs, (six.string_types, six.text_type)):
            inputs = [inputs]
        if isinstance(outputs, (six.string_types, six.text_type)):
            outputs = [outputs]
        if not isinstance(inputs, list) or not all(isinstance(s, (six.string_types, six.text_type)) for s in inputs):
            type_list = ','.join(list(str(type(s)) for s in inputs))
            raise ValueError('Inputs must be a list of string but get [%s]' % type_list)
        if not isinstance(outputs, list) or not all(isinstance(s, (six.string_types, six.text_type)) for s in outputs):
            type_list = ','.join(list(str(type(s)) for s in outputs))
            raise ValueError('Outputs must be a list of string but get [%s]' % type_list)
        for k, v in attrs.items():
            if v is None:
                raise ValueError('Failed to create ONNX node. Undefined attribute pair (%s, %s) found' % (k, v))

        if 'name' in attrs.keys():
            if attrs['name'] in self.node_names:
                cur_count = self.node_names[attrs['name']] + 1
                self.node_names.update({attrs['name']: cur_count})
                attrs['name'] = attrs['name'] + "_" + str(cur_count)
            self.node_names.update({attrs['name']: 0})

        node = helper.make_node(op_type, inputs, outputs, **attrs)
        node.domain = op_domain

        self.node_domain_version_pair_sets.add((op_domain, op_version))
        self.nodes.append(node)

    def add_onnx_node(self, proto_node, op_domain='', op_version=1):
        self.node_domain_version_pair_sets.add((op_domain, op_version))
        self.nodes.append(proto_node)


class InterimContext:
    """
    The interim context was kept to re-use the keras converter code from onnxmltools, which has
    an input parameter called as 'scope'.
    """

    def __init__(self, name, parent_scopes=None, variable_name_set=None, operator_name_set=None,
                 target_opset=None):
        """
        :param name:  A string, the unique ID of this scope in a Topology object
        :param parent_scopes: A list of Scope objects. The last element should be the direct parent scope (i.e., where
        this scope is declared).
        :param variable_name_set: A set of strings serving as the name pool of variables
        :param operator_name_set: A set of strings serving as the name pool of operators
        :param targeted_onnx_version: A StrictVersion object indicating the ONNX version used
        """
        self.name = name
        self.parent_scopes = parent_scopes if parent_scopes else list()
        self.onnx_variable_names = variable_name_set if variable_name_set is not None else set()
        self.onnx_operator_names = operator_name_set if operator_name_set is not None else set()
        self.target_opset = target_opset

        # An one-to-many map from raw variable name to ONNX variable names. It looks like
        #   (key, value) = (raw_name, [onnx_name, onnx_name1, onnx_name2, ..., onnx_nameN])
        # The last name may hide all other names in this scope.
        self.variable_name_mapping = {}

        # A map of local variables defined in this scope. (key, value) = (onnx_name, variable)
        self.variables = {}

        # A map of local operators defined in this scope. (key, value) = (onnx_name, operator)
        self.operators = {}

        self.prefix = name + '/'

    def get_onnx_variable_name(self, seed):
        """
        Retrieve the variable ID of the given seed or create one if it is the first time of seeing this seed
        """
        if seed in self.variable_name_mapping:
            return self.variable_name_mapping[seed][-1]
        else:
            return self.get_unique_variable_name(seed)

    def get_unique_variable_name(self, seed):
        """
        Create a unique variable ID based on the given seed
        """
        if seed.startswith(self.prefix):
            seed = seed[len(self.prefix):]
        return Variable.generate_unique_name(seed, self.onnx_variable_names)

    def get_unique_operator_name(self, seed):
        """
        Create a unique operator ID based on the given seed
        """
        if seed.startswith(self.prefix):
            seed = seed[len(self.prefix):]
        return Variable.generate_unique_name(seed, self.onnx_operator_names)

    def find_sink_variables(self):
        """
        Find sink variables in this scope
        """
        # First we assume all variables are sinks
        is_sink = {name: True for name in self.variables.keys()}
        # Then, we remove those variables which are inputs of some operators
        for operator in self.operators.values():
            for variable in operator.inputs:
                is_sink[variable.onnx_name] = False
        return [variable for name, variable in self.variables.items() if is_sink[name]]

    def declare_local_variable(self, raw_name, type=None, prepend=False):
        """
        This function may create a new variable in this scope. If raw_name has been used to create other variables,
        the new variable will hide all other variables created using raw_name.
        """
        # Get unique ID for the new variable
        onnx_name = self.get_unique_variable_name(raw_name)

        # Create the variable
        variable = Variable(raw_name, onnx_name, self.name, type)
        self.variables[onnx_name] = variable

        if raw_name in self.variable_name_mapping:
            # Hide existing variables with the same raw_name
            if not prepend:
                self.variable_name_mapping[raw_name].append(onnx_name)
            else:
                self.variable_name_mapping[raw_name].insert(0, onnx_name)
        else:
            self.variable_name_mapping[raw_name] = [onnx_name]
        return variable

    def get_local_variable_or_declare_one(self, raw_name, type=None):
        """
        This function will first check if raw_name has been used to create some variables. If yes, the latest one
        named in self.variable_name_mapping[raw_name] will be returned. Otherwise, a new variable will be created and
        then returned.
        """
        onnx_name = self.get_onnx_variable_name(raw_name)
        if onnx_name in self.variables:
            return self.variables[onnx_name]
        else:
            variable = Variable(raw_name, onnx_name, self.name, type)
            self.variables[onnx_name] = variable
            if raw_name in self.variable_name_mapping:
                self.variable_name_mapping[raw_name].append(onnx_name)
            else:
                self.variable_name_mapping[raw_name] = [onnx_name]
            return variable

    def declare_local_operator(self, type, raw_model=None, op_name=None, **attrs):
        """
        This function is used to declare new local operator.
        """
        onnx_name = self.get_unique_operator_name(str(type) if op_name is None else op_name)
        operator = Operator(onnx_name, self.name, type, raw_model, self.target_opset)
        self.operators[onnx_name] = operator
        operator.update_attrs(**attrs)
        return operator

    def delete_local_operator(self, onnx_name):
        """
        Remove the operator whose onnx_name is the input onnx_name
        """
        if onnx_name not in self.onnx_operator_names or onnx_name not in self.operators:
            raise RuntimeError('The operator to be removed not found')
        self.onnx_operator_names.discard(onnx_name)
        del self.operators[onnx_name]

    def delete_local_variable(self, onnx_name):
        """
        Remove the variable whose onnx_name is the input onnx_name
        """
        if onnx_name not in self.onnx_variable_names or onnx_name not in self.variables:
            raise RuntimeError('The variable to be removed not found')
        self.onnx_variable_names.discard(onnx_name)
        raw_name = self.variables[onnx_name].raw_name
        self.variable_name_mapping[raw_name].remove(onnx_name)
        del self.variables[onnx_name]


class Variable:
    """
    The tensor or other data types
    """

    def __init__(self, raw_name, onnx_name, scope, type=None):
        """
        :param raw_name: A string indicating the variable's name in the original model. Usually, it's the seed string
        used to created its ONNX name (i.e., the field onnx_name below).
        :param onnx_name: A string indicating the variable's name in the converted model
        :param scope: A string. It's the name of the scope where this variable is declared
        :param type: A type object defined in onnxmltools.convert.common.data_types.py; e.g., FloatTensorType
        """
        self.raw_name = raw_name  #
        self.onnx_name = onnx_name  #
        self.scope = scope
        self.type = type
        # The following fields are bool variables used in parsing and compiling stages
        self.is_fed = None
        self.is_root = None
        self.is_leaf = None
        self.is_abandoned = False
        self.op_from = None
        self.op_to = []

    @property
    def full_name(self):
        """
        Return a globally unique variable ID
        """
        return self.onnx_name

    def __repr__(self):
        return self.onnx_name

    @staticmethod
    def tfname_to_onnx(name):
        # tf2onnx does not change name but still keep '/'.
        # We should not modify name here, otherwise it causes issues in subgraph in operators.
        return name

    @staticmethod
    def generate_unique_name(seed, existing_names):
        """
        Produce an unique string based on the seed
        :param seed: a string
        :param existing_names: a set containing strings which cannot be produced
        :return: a string similar to the seed
        """
        if seed == '':
            raise ValueError('Name seed must be an non-empty string')

        # Make the seed meet C-style naming convention
        seed = Variable.tfname_to_onnx(seed)

        # If seed has never been seen, we return it as it is. Otherwise, we will append an number to make it unique.
        if seed not in existing_names:
            existing_names.add(seed)
            return seed
        else:
            i = 1
            while seed + str(i) in existing_names:
                i += 1
            new_name = seed + str(i)
            existing_names.add(new_name)
            return new_name
