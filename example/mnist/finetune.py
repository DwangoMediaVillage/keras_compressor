from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras_compressor.layers import custom_layers


def preprocess(X):
    return X.astype('float32') / 255


class_num = 10
batch_size = 128
epochs = 100

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

model = load_model('model_compressed.h5', custom_layers)
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'],
)

model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[
        EarlyStopping(patience=3),
    ],
)

score = model.evaluate(X_test, y_test)

print('test accuracy: ', score[1])

# re-compile model
# not to save optimizer variables in model data
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'],
)
model.save('model_finetuned.h5')
