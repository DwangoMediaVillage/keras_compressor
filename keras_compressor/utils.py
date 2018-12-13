from typing import Any, Dict, List

from keras.engine.topology import Layer, Node


def swap_layer_connection(old_layer: Layer, new_layer: Layer) -> None:
    '''connect nodes of calc graph for new_layer and disconnect ones for old_layers

    Keras manages calculation graph by nodes which hold connection between
    layres. To swap old layer and new layer, it is required to delete nodes
    of old layer and to create new nodes of new layer.

    :arg old_layer: Old layer. The connection to/from this layer will be removed.
    :arg new_layer: New layer. The connection to/from old layer will be connected to/from
        this layer.
    :return: None
    '''

    # the set of inbound layer which have old outbound_node
    inbound_layers = set()

    # create new inbound nodes
    for node in old_layer._inbound_nodes:  # type: Node
        Node(
            new_layer, node.inbound_layers,
            node.node_indices, node.tensor_indices,
            node.input_tensors, node.output_tensors,
            node.input_masks, node.output_masks,
            node.input_shapes, node.output_shapes,
        )
        inbound_layers.union(set(node.inbound_layers))

    # remove old outbound node of inbound layers
    for layer in inbound_layers:  # type: Layer
        old_nodes = filter(
            lambda n: n.outbound_layer == old_layer,
            layer._outbound_nodes,
        )
        for n in old_nodes:  # type: Node
            layer._outbound_nodes.remove(n)

    # the set of outbound layer which have old inbound_nodes
    outbound_layers = set()
    # create new outbound nodes
    for node in old_layer._outbound_nodes:  # type: Node
        layers = list(node.inbound_layers)
        while old_layer in layers:
            idx = layers.index(old_layer)
            layers[idx] = new_layer
        Node(
            node.outbound_layer, layers,
            node.node_indices, node.tensor_indices,
            node.input_tensors, node.output_tensors,
            node.input_masks, node.output_masks,
            node.input_shapes, node.output_shapes,
        )
        outbound_layers.add(node.outbound_layer)

    # remove old inbound_node of outbound layers
    for layer in outbound_layers:  # type: Layer
        old_nodes = filter(
            lambda n: old_layer in n.inbound_layers,
            layer._inbound_nodes,
        )
        for n in old_nodes:
            layer._inbound_nodes.remove(n)


def convert_config(
        base_config: Dict[str, Any],
        ignore_args: List[str],
        converts: Dict[str, List[str]],
        new_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    '''convert old layer's config to new layer's config.

    :param base_config: Base config. Generally a config of old layer.
    :param ignore_args: Ignore arg names. Not required arg names in new layer,
        though them is required in old layer.
    :param converts: Ignore name conversion dictionary, whose key is old layer's
        arg name in base_config, and whose value is new layer's arg names(list).
    :param new_kwargs: The new kwargs.
    :return: Converted config.
    '''
    kwargs = {}
    for k, v in base_config.items():
        if k in ignore_args:
            continue
        elif k in converts:
            for new_k in converts[k]:
                kwargs[new_k] = v
        else:
            kwargs[k] = v
    kwargs.update(new_kwargs)
    return kwargs
