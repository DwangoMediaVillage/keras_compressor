import sys

from keras import backend as K
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
from keras_compressor import custom_objects


def preprocess(X):
    return X.astype('float32') / 255


def usage():
    print('{} model.h5'.format(sys.argv[0]))


def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = preprocess(X_train), preprocess(X_test)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    _, img_rows, img_cols = X_train.shape

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    return (X_train, y_train), (X_test, y_test)


def main():
    if len(sys.argv) != 2:
        usage()
        sys.exit(1)

    model_path = sys.argv[1]
    _, (X_test, y_test) = load_mnist()

    model = load_model(model_path, custom_objects)
    result = model.evaluate(X_test, y_test, verbose=0)
    model.summary()
    print(result)


if __name__ == '__main__':
    main()
