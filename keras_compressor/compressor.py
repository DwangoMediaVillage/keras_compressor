import logging
from collections import defaultdict
from typing import Dict, List, Type

from keras.engine import Layer, Model

from .factorizer import Factorizer
from .factorizers.svd import SVDFactorizer
from .factorizers.tucker import TuckerFactorizer
from .utils import swap_layer_connection

logger = logging.getLogger(__name__)


def compress(model: Model, acceptable_error: float,
             factorizers=None) -> Model:
    """compress model under acceptable error

    compress each model's layer by using given factorizers.
    If the factorizer compress the layer, swap the layer and compressed layer by re-creating
    node on computational graph.

    :param model: Target model
    :param acceptable_error: Layer-wize acceptable output error. If this value is smaller
        the compressed model will be more accurate. The calculation process of this error is
        depend on each factorizer. So see the implementation.
    :param factorizers: Applicable factorizers. Factorizer factorize each layer if factorizer
        can factorize the layer.
    :return: Compressed model
    """
    if factorizers is None:
        factorizers = [SVDFactorizer, TuckerFactorizer]
    layer2factorizers = defaultdict(list)  # type: Dict[Type[Layer], List[Type[Factorizer]]]
    for fact in factorizers:
        for layer in fact.factorize_target_layers:
            layer2factorizers[layer].append(fact)

    for layer_idx, layer in enumerate(model.layers):
        layer_class = type(layer)
        if layer_class not in layer2factorizers:
            logger.info(
                'factorizer not found layer:{!r}'.format(layer)
            )
            continue

        new_layer = None
        for factorizer in layer2factorizers[layer_class]:  # type: Factorizer
            logger.info(
                'factorizer found layer:{!r} factorizer:{!r}'.format(
                    layer, factorizer,
                )
            )
            new_layer = factorizer.compress(layer, acceptable_error)
            if new_layer is None:  # failed factorization
                logger.info(
                    'factorization failed layer:{!r} factorizer:{!r}'.format(
                        layer, factorizer,
                    )
                )
                continue
            else: # succeeded factorization
                break

        if new_layer is not None:
            logger.info(
                'swap old/new layer old_layer:{!r} new_layer{!r}'.format(
                    layer, new_layer,
                )
            )
            swap_layer_connection(layer, new_layer)
            model.layers[layer_idx] = new_layer

    new_model = Model(model.inputs, model.outputs)
    new_model.compile(
        optimizer=model.optimizer.__class__.__name__,  # TODO: improve here
        # model.optimizer is instance of Optimizer and hold some variables for target model.
        # Optimizer must be re-initialized, because compress function changes model structure.
        loss=model.loss,
        metrics=model.metrics,
        loss_weights=model.loss_weights,
        sample_weight_mode=model.sample_weight_mode,
    )
    return new_model
