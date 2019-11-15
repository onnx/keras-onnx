###############################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
from ..proto import keras, is_tf_keras, is_keras_older_than


class LayerInfo(object):
    def __init__(self, _ly):
        self.layer = _ly
        self.inputs = []
        self.outputs = []
        self.nodelist = []

    @staticmethod
    def create(layer, outputs_map, inference_nodeset):
        layer_info = LayerInfo(layer)
        # find the output
        visited = set()
        for ts_, layer_info_ in outputs_map.items():
            if layer_info_[0] == layer:
                visited.add(ts_.op)
                layer_info.outputs.append(ts_)

        next_itr = set(visited)
        while next_itr:
            visited |= next_itr
            next_itr.clear()
            for n_ in visited:
                for i_ in n_.inputs:
                    if i_ in outputs_map or i_.op in visited or i_.op not in inference_nodeset:
                        continue
                    next_itr.add(i_.op)

        layer_info.nodelist = list(visited)
        return layer_info


def tsname_to_node(name):
    return name.split(':')[0]


def is_subclassing(model):
    return not (model._is_graph_network or  # pylint:disable=protected-access
                isinstance(model, keras.engine.sequential.Sequential))


def _find_output(outputs, name):
    for ou_ in outputs:
        if ou_.name.find('/{}/'.format(name)) >= 0:
            return ou_

    return None


def build_layer_outputs(model, graph, outputs):
    # type: (keras.Model, []) -> {}

    output_dict = {}
    for l_ in model.layers:
        if hasattr(l_, 'layers'):
            submodel_dict = build_layer_outputs(l_, graph, outputs)
            output_dict.update(submodel_dict)
            continue

        ts = _find_output(outputs, l_.name)
        if ts is not None:
            assert graph.get_operation_by_name(tsname_to_node(ts.name)), \
                "Cannot find the {} in the graph".format(ts.name)
            output_dict[ts.name] = (l_, model)

    return output_dict


def outputs_to_dict(graph, outputs):
    t2l_dict = {}
    for k_, v_ in outputs.items():
        op = graph.get_operation_by_name(tsname_to_node(k_))
        assert op is not None, "Cannot find the {} in the graph".format(k_)
        t2l_dict.update({ts_k_: v_ for ts_k_ in op.outputs})

    return t2l_dict
