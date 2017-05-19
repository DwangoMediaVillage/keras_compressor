import itertools
import logging
import math
from queue import PriorityQueue
from typing import Optional, Tuple

import numpy as np
from keras import backend as K
from keras.layers import Conv2D
from keras_compressor.factorizer import Factorizer
from keras_compressor.layers import FactorizedConv2DTucker
from keras_compressor.utils import convert_config
from sklearn.utils.extmath import randomized_svd

logger = logging.getLogger(__name__)

__all__ = ['TuckerFactorizer']


class ProblemData:
    '''Parameter search problem data structure
    '''

    def __init__(self, x_range: range, y_range: range):
        self.x_range = x_range
        self.y_range = y_range

    def __str__(self):
        return '<Problem x_range={} y_range={}>'.format(
            self.x_range, self.y_range,
        )

    def __lt__(self, other: 'ProblemData'):
        return self.diag_length < other.diag_length

    def __eq__(self, other: 'ProblemData'):
        return self.x_range == other.x_range and self.y_range == other.y_range

    @property
    def diag_length(self):
        return math.sqrt(len(self.x_range) ** 2 + len(self.y_range) ** 2)


class Tucker:
    '''Pure tucker decomposition functions
    '''

    @classmethod
    def factorize(cls, W: np.ndarray, in_comps: Optional[int], out_comps: Optional[int]) \
            -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """pure tucker decomposition

        :param W: W x H x I x O
        :param in_comps: N
        :param out_comps:  M
        :return:
            C: W x H x N x M,
            U_in: I x N
            U_out: O x M
        """

        if in_comps is None:
            U_in = None
        else:
            U_in, _, _ = randomized_svd(cls._flatten(W, 2), in_comps)
            U_in = U_in.astype(W.dtype)

        if out_comps is None:
            U_out = None
        else:
            U_out, _, _ = randomized_svd(cls._flatten(W, 3), out_comps)
            U_out = U_out.astype(W.dtype)

        C = W.copy()

        if U_in is not None:
            C = np.einsum('whio,in->whno', C, U_in)

        if U_out is not None:
            C = np.einsum('whno,om->whnm', C, U_out)

        C = C.astype(W.dtype)

        return C, U_in, U_out

    @staticmethod
    def _get_matrix(W: np.ndarray, i: int, axis: int) -> np.ndarray:
        '''util function for tucker decomposition

        :param W:
        :param i:
        :param axis:
        :return:
        '''
        sli = [slice(None) for _ in range(W.ndim)]
        sli[axis] = i
        return W[sli]

    @classmethod
    def _flatten(cls, W: np.ndarray, axis: int) -> np.ndarray:
        '''util function for tucker decomposition

        :param W:
        :param axis:
        :return:
        '''
        dim = 1
        dims = []
        for i, v in enumerate(W.shape):
            if i != axis:
                dim *= v
                dims.append(v)
        res = np.zeros((W.shape[axis], dim))
        for i in range(W.shape[axis]):
            res[i] = cls._get_matrix(W, i, axis).ravel()
        return res


class TuckerParamSearcher:
    '''tucker decomposition parameter searcher
    '''

    def __init__(self, W: np.ndarray):
        width, height, in_dim, out_dim = W.shape
        self.width = width
        self.height = height
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = W

        self.best_point = None
        self.best_param_num = width * height * in_dim * out_dim

        self.prob_queue = PriorityQueue()

    def add_problem(self, prob: ProblemData):
        param_num = self.calc_min_param_num_by(prob)
        self.prob_queue.put((param_num, prob))

    def calc_min_param_num_by(self, prob: ProblemData):
        res = []
        for in_comp, out_comp in itertools.product(
                [prob.x_range.start, prob.x_range.stop],
                [prob.y_range.start, prob.y_range.stop],
        ):
            res.append(self.calc_param_num(in_comp, out_comp))
        return min(res)

    def calc_param_num(self, in_comp: int, out_comp: int):
        params = self.width * self.height * in_comp * out_comp
        if in_comp != self.in_dim:  # compression in input channel
            params += self.in_dim * in_comp
        if out_comp != self.out_dim:  # compression in output channel
            params += self.out_dim * out_comp
        return params

    def update_best_point_if_needed(self, in_comp, out_comp):
        current_param_num = self.calc_param_num(in_comp, out_comp)
        if current_param_num < self.best_param_num:
            self.best_point = (in_comp, out_comp)
            self.best_param_num = current_param_num
            logger.debug('update best_point={} best_param_num={}'.format(
                self.best_point, self.best_param_num,
            ))

    def factorize_in_acceptable_error(self, acceptable_error: float) \
            -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        :param W: W x H x I x O
        :param acceptable_error:
        :param initial_problems: initial problems
        :return:
            C: W x H x N x M
            U_in: I x N
            U_out: O x M
        """

        # Search N and M whose the number of parameter is the smallest
        # under acceptable error.

        # This algorithm is based on divide and conquer algorithm and
        # based on the assumption that `error` monotonically decrease by increasing N or M.

        width, height, in_dim, out_dim = self.width, self.height, self.in_dim, self.out_dim

        self.add_problem(ProblemData(range(in_dim, in_dim), range(out_dim, out_dim)))
        if 2 <= in_dim and 2 <= out_dim:
            self.add_problem(ProblemData(range(1, in_dim - 1), range(1, out_dim - 1)))
        if 1 <= in_dim:
            self.add_problem(ProblemData(range(in_dim, in_dim), range(1, out_dim - 1)))
        if 1 <= out_dim:
            self.add_problem(ProblemData(range(1, in_dim - 1), range(out_dim, out_dim)))

        while not self.prob_queue.empty():
            _, current_prob = self.prob_queue.get()  # type: ProblemData

            if self.best_param_num < self.calc_min_param_num_by(current_prob):
                logger.debug('no more best_param_num:{} prob:{}'.format(self.best_param_num, current_prob))
                break

            if len(current_prob.x_range) == 0 and len(current_prob.y_range) == 0:
                self.update_best_point_if_needed(current_prob.x_range.start, current_prob.y_range.start)
                continue

            logger.debug('current queue.size:{} prob:{}'.format(
                self.prob_queue.qsize(), current_prob,
            ))

            result = self._find_edge_point(
                acceptable_error, current_prob,
            )  # type: Optional[Tuple[int,int]]

            logger.debug('result={} prob={}'.format(
                result, current_prob
            ))

            if result is None:
                continue

            x, y = result  # type: Tuple[int, int]
            self.update_best_point_if_needed(x, y)

            # divide current problem to sub-problems

            if len(current_prob.x_range) == 0 or len(current_prob.y_range) == 0:
                # X.
                #  Y
                logger.debug('no sub-problem:{}'.format(current_prob))
                continue

            if len(current_prob.x_range) == 1 and len(current_prob.y_range) == 1:
                # X# -> (| and _) or .
                #  Y
                if x == current_prob.x_range.stop and y == current_prob.y_range.stop:
                    # right top point
                    #        _
                    # X# -> X  and  |
                    #  Y     Y     Y
                    sub_prob1 = ProblemData(
                        x_range=range(current_prob.x_range.start, x),
                        y_range=range(y, y),
                    )
                    sub_prob2 = ProblemData(
                        x_range=range(x, x),
                        y_range=range(current_prob.y_range.start, y),
                    )
                    self.add_problem(sub_prob1)
                    self.add_problem(sub_prob2)
                    logger.debug('two sub-problems:{}, (x,y)=({},{}) -> {},{}'.format(
                        current_prob, x, y,
                        sub_prob1, sub_prob2
                    ))
                    continue
                else:  # x == current_prob.x_range.start and y == current_prob.y_range.start
                    logger.debug('no sub-problem:{}'.format(current_prob))
                    continue

            if len(current_prob.x_range) == 1 and len(current_prob.y_range) > 1:
                # X####### -> X   |
                #     Y          Y
                sub_prob = ProblemData(
                    x_range=current_prob.x_range,
                    y_range=range(y, y)
                )
                logger.debug('one row space, one sub-problem:{}, (x,y)=({},{}) -> {}'.format(
                    current_prob, x, y,
                    sub_prob,
                ))
                continue

            if len(current_prob.x_range) > 1 and len(current_prob.y_range) == 1:
                #  #     _
                # X# -> X
                #  #     Y
                #  Y
                sub_prob = ProblemData(
                    x_range=range(x, x),
                    y_range=current_prob.y_range,
                )
                logger.debug('one column space, one sub-problem:{}, (x,y)=({},{}) -> {}'.format(
                    current_prob, x, y,
                    sub_prob,
                ))

            if len(current_prob.x_range) >= 2 and len(current_prob.y_range) >= 2:
                #  ###     ##
                # X### -> X##  and X
                #  ###                #
                #   Y       Y        Y
                sub_prob1 = ProblemData(
                    x_range=range(current_prob.x_range.start, x),
                    y_range=range(y, current_prob.y_range.stop),
                )
                sub_prob2 = ProblemData(
                    x_range=range(x, current_prob.x_range.stop),
                    y_range=range(current_prob.y_range.start, y),
                )
                self.add_problem(sub_prob1)
                self.add_problem(sub_prob2)
                logger.debug('two sub-problems:{}, (x,y)=({},{}) -> {},{}'.format(
                    current_prob, x, y,
                    sub_prob1, sub_prob2
                ))
                continue
        if self.best_point is None:
            logger.debug('no factorization is best')
            return self.W, None, None

        in_comp, out_comp = self.best_point
        if in_comp >= self.in_dim:
            in_comp = None
        if out_comp >= self.out_dim:
            out_comp = None
        C, U_in, U_out = Tucker.factorize(self.W, in_comp, out_comp)
        return C, U_in, U_out

    def _find_edge_point(self, acceptable_error: float, current_prob: ProblemData) -> Optional[Tuple[int, int]]:
        x_range = current_prob.x_range
        y_range = current_prob.y_range

        acceptable_points = []
        # consider that acceptable point doesn't exist in the current_prob space.

        while len(x_range) > 0 or len(y_range) > 0:
            if len(x_range) in [0, 1]:
                x = x_range.start
            else:
                x = round((x_range.start + x_range.stop) / 2)
            if len(y_range) in [0, 1]:
                y = y_range.start
            else:
                y = round((y_range.start + y_range.stop) / 2)

            logger.debug('binary search (x,y)=({}, {}) x_range={} y_range={} prob={}'.format(
                x, y, x_range, y_range, current_prob
            ))

            C, U_in, U_out = Tucker.factorize(self.W, x, y)

            error = self.calc_error(self.W, C, U_in, U_out)
            if error < acceptable_error:
                logger.debug('binary search: under threshold={} error={}'.format(
                    acceptable_error, error,
                ))
                acceptable_points.append((x, y))

                # update ranges
                x_range = range(x_range.start, x)
                y_range = range(y_range.start, y)
            else:
                logger.debug('binary search: over threshold={} error={}'.format(
                    acceptable_error, error,
                ))

                # update ranges
                if x + 1 <= x_range.stop:
                    new_x_start = x + 1
                else:
                    new_x_start = x
                x_range = range(new_x_start, x_range.stop)

                if y + 1 <= y_range.stop:
                    new_y_start = y + 1
                else:
                    new_y_start = y
                y_range = range(new_y_start, y_range.stop)
        if len(acceptable_points) == 0:
            return None
        else:
            return acceptable_points[-1]

    @staticmethod
    def calc_error(W: np.ndarray, C: np.ndarray, U_in: np.ndarray, U_out: np.ndarray) -> float:
        """calculate expected bound of error of output of layer

        :param W: W x H x I x O
        :param C: W x H x N x M
        :param U_in: I x N
        :param U_out: O x M
        :return:
        """
        W_hat = np.einsum('whnm,in,om->whio', C, U_in, U_out)
        elemental_error = np.abs(W - W_hat)
        error_bound = np.mean(elemental_error) / np.mean(np.abs(W))
        return error_bound


class TuckerFactorizer(Factorizer):
    factorize_target_layers = [Conv2D]

    @classmethod
    def compress(cls, old_layer: Conv2D, acceptable_error: float) -> Optional[FactorizedConv2DTucker]:
        '''Compress layer's kernel 4D tensor using tucker decomposition under acceptable_error.

        If it can't reduce the number of parameters, returns `None`.

        :param old_layer:
        :param acceptable_error:
        :return:
        '''
        W = K.get_value(old_layer.kernel)
        searcher = TuckerParamSearcher(W)
        C, U_in, U_out = searcher.factorize_in_acceptable_error(acceptable_error)

        kernel = C

        if U_in is None and U_out is None:  # compression failed
            return None

        if U_in is None:
            input_components = None
            pre_kernel = None
        else:
            input_components = U_in.shape[1]
            pre_kernel = U_in[np.newaxis, np.newaxis, :, :]

        if U_out is None:
            output_components = None
            post_kernel = None
        else:
            output_components = U_out.shape[1]
            post_kernel = U_out.T[np.newaxis, np.newaxis, :, :]

        base_config = old_layer.get_config()

        new_config = convert_config(
            base_config,
            ignore_args=[
                'kernel_constraint',
            ],
            converts={
                'kernel_regularizer': [
                    'pre_kernel_regularizer',
                    'kernel_regularizer',
                    'post_kernel_regularizer',
                ],
                'kernel_initializer': [
                    'pre_kernel_initializer',
                    'kernel_initializer',
                    'post_kernel_initializer',
                ],
            },
            new_kwargs={
                'input_components': input_components,
                'output_components': output_components,
            }
        )

        new_layer = FactorizedConv2DTucker(**new_config)
        new_layer.build(old_layer.get_input_shape_at(0))  # to initialize weight variables

        K.set_value(new_layer.kernel, kernel)

        if pre_kernel is not None:
            K.set_value(new_layer.pre_kernel, pre_kernel)
        if post_kernel is not None:
            K.set_value(new_layer.post_kernel, post_kernel)
        return new_layer
