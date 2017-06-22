#!/usr/bin/env python
import argparse
import logging

import keras
import keras.backend as K
import numpy
from keras.models import load_model

from keras_compressor.compressor import compress


def count_total_params(model):
    """Counts the number of parameters in a model

    See:
        https://github.com/fchollet/keras/blob/172397ebf45d58ba256c10004c6fce8b40df286b/keras/utils/layer_utils.py#L114-L117

    :param model: Keras model instance
    :return: trainable_count, non_trainable_count
    :rtype: tuple of int
    """
    trainable_count = int(
        numpy.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        numpy.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    return trainable_count, non_trainable_count


def gen_argparser():
    parser = argparse.ArgumentParser(description='compress keras model')
    parser.add_argument('model', type=str, metavar='model.h5',
                        help='target model, whose loss is specified by `model.compile()`.')
    parser.add_argument('compressed', type=str, metavar='compressed.h5',
                        help='compressed model path')
    parser.add_argument('--error', type=float, default=0.1, metavar='0.1',
                        help='layer-wise acceptable error. '
                             'If this value is larger, compressed model will be '
                             'less accurate and achieve better compression rate. '
                             'Default: 0.1')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                        help='log level. Default: INFO')
    return parser


def main():
    parser = gen_argparser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    model = load_model(args.model)  # type: keras.models.Model
    total_params_before = sum(count_total_params(model))
    model = compress(model, acceptable_error=args.error)
    total_params_after = sum(count_total_params(model))
    model.save(args.compressed)
    print('\n'.join((
        'Compressed model',
        '    before #params {:>20,d}',
        '    after  #params {:>20,d} ({:.2%})',
    )).format(
        total_params_before, total_params_after, 1 - float(total_params_after) / total_params_before,
    ))


if __name__ == '__main__':
    main()
