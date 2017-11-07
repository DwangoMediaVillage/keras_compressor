# keras_compressor
Model compression CLI tool for [keras](https://github.com/fchollet/keras).

# How to use it

## Requirements
- Python 3.5, 3.6
- Keras
    - We tested on Keras 2.0.3 (TensorFlow backend)

## Install
```
$ git clone ${this repository}
$ cd ./keras_compressor
$ pip install .
```

## Compress
Simple example:
```
$ keras-compressor.py model.h5 compressed.h5
```

With accuracy parameter `error`:
```
$ keras-compressor.py --error 0.001 model.h5 compressed.h5
```

## Help
```
$ keras-compressor.py --help                                                                               [impl_keras_compressor:keras_compressor]
Using TensorFlow backend.
usage: keras-compressor.py [-h] [--error 0.1]
                           [--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}]
                           model.h5 compressed.h5

compress keras model

positional arguments:
  model.h5              target model, whose loss is specified by
                        `model.compile()`.
  compressed.h5         compressed model path

optional arguments:
  -h, --help            show this help message and exit
  --error 0.1           layer-wise acceptable error. If this value is larger,
                        compressed model will be less accurate and achieve
                        better compression rate. Default: 0.1
  --log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}
                        log level. Default: INFO
```

# How compress it
- low-rank approximation
  - with SVD (matrix)
  - with Tucker (tensor)

# Examples
In example directory, you will find model compression of VGG-like models using MNIST and CIFAR10 dataset.

```console
$ cd ./keras_compressor/example/mnist/

$ python train.py
-> outputs non-compressed model `model_raw.h5`

$ python compress.py
-> outputs compressed model `model_compressed.h5` from `model_raw.h5`

$ python finetune.py
-> outputs finetuned and compressed model `model_finetuned.h5` from `model_compressed.h5`

$ python evaluate.py model_raw.h5
$ python evaluate.py model_compressed.h5
$ python evaluate.py model_finetuned.h5
-> output test accuracy and the number of model parameters
```
