from typing import List, Optional, Type

from keras.layers import Layer


class Factorizer:
    factorize_target_layers = []  # type: List[Type[Layer]]

    @classmethod
    def compress(cls, layer: Layer, acceptable_error: float) -> Optional[Layer]:
        """try to compress the layer under acceptable_error.

        Outputs compressed layer if compression succeeded. If not, return None.

        :param layer:
        :param acceptable_error:
        :return:
        """
        raise NotImplementedError
