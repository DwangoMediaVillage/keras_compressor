import logging
import math
from typing import Optional, Tuple

import numpy as np
from keras import backend as K
from keras.engine import Layer
from keras.layers import Dense
from sklearn.utils.extmath import randomized_svd

from ..factorizer import Factorizer
from ..layers import FactorizedDense
from ..utils import convert_config

logger = logging.getLogger(__name__)


class SVDFactorizer(Factorizer):
    factorize_target_layers = [Dense]

    @staticmethod
    def _factorize(W: np.ndarray, components: int):
        """
        :param W: I x O
        :param components: K
        :return:
            U: I x K
            V: K x O
        """
        u, s, v = randomized_svd(W, components)
        scale = np.diag(np.sqrt(s))
        U, V = u.dot(scale).astype(W.dtype), scale.dot(v).astype(W.dtype)
        return U, V

    @staticmethod
    def _calc_error(W: np.ndarray, U: np.ndarray, V: np.ndarray):
        '''
        :param W: I x O
        :param U: I x K
        :param V: K x O
        :return:
        '''
        elemental_error = np.abs(W - U.dot(V))
        error_bound = np.mean(elemental_error) / np.mean(np.abs(W))
        return error_bound

    @classmethod
    def compress(cls, old_layer: Dense, acceptable_error: float) -> Optional[Layer]:
        '''compress old_layer under acceptable error using SVD.

        If it can't reduce the number of parameters, returns None,

        :param old_layer:
        :param acceptable_error:
        :return:
        '''
        W = K.get_value(old_layer.kernel)
        logger.debug('factorization start W.shape:{}'.format(W.shape))

        max_comps = math.floor(np.size(W) / sum(W.shape))
        U, V = cls._factorize(W, max_comps)
        if cls._calc_error(W, U, V) >= acceptable_error:
            # Factorizer can't reduce the number of parameters in acceptable error by SVD.
            # So, this factorizer failed compression.
            return None

        U, V = cls._compress_in_acceptable_error(
            W, acceptable_error,
            start_param_range=range(1, max_comps),
        )
        components = U.shape[-1]

        base_config = old_layer.get_config()

        new_config = convert_config(
            base_config,
            ignore_args=[
                'kernel_constraint',
            ],
            converts={
                'kernel_regularizer': [
                    'pre_kernel_regularizer',
                    'post_kernel_regularizer',
                ],
                'kernel_initializer': [
                    'pre_kernel_initializer',
                    'post_kernel_initializer',
                ]
            },
            new_kwargs={
                'components': components,
            },
        )

        new_layer = FactorizedDense(**new_config)
        new_layer.build(old_layer.get_input_shape_at(0))  # to initialize weight variables

        K.set_value(new_layer.pre_kernel, U)
        K.set_value(new_layer.post_kernel, V)

        return new_layer

    @classmethod
    def _compress_in_acceptable_error(cls, W, acceptable_error: float, start_param_range: range) \
            -> Tuple[np.ndarray, np.ndarray]:
        param_range = start_param_range
        while len(param_range) > 0:  # while not (param_range.start == param_range.stop)
            logger.debug('current param_range:{}'.format(param_range))
            if len(param_range) == 1:
                ncomp = param_range.start
            else:
                ncomp = round((param_range.start + param_range.stop) / 2)

            U, V = cls._factorize(W, ncomp)
            error = cls._calc_error(W, U, V)

            if error <= acceptable_error:  # smallest ncomp is equal to or smaller than ncomp
                # On the assumption that `error` monotonically decreasing by increasing ncomp
                logger.debug('under acceptable error ncomp:{} threshold:{} error:{}'.format(
                    ncomp, acceptable_error, error,
                ))
                param_range = range(param_range.start, ncomp)
            else:  # the best is larger than ncomp
                logger.debug('over acceptable error ncomp:{} threshold:{} error:{}'.format(
                    ncomp, acceptable_error, error,
                ))
                param_range = range(ncomp + 1, param_range.stop)

        # param_range.start == param_range.stop
        smallest_ncomp = param_range.start
        logger.debug('smallest_ncomp:{}, W.shape:{} compress_rate:{}'.format(
            smallest_ncomp, W.shape, sum(W.shape) * smallest_ncomp / np.size(W),
        ))
        U, V = cls._factorize(W, smallest_ncomp)
        return U, V
