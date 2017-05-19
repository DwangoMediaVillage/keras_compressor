import keras.backend as K
import keras.callbacks as C
from keras.datasets import cifar10
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras_compressor import custom_objects

def preprocess(X):
    return X.astype('float32') / 255 * 2 - 1


class_num = 10
batch_size = 128
epochs = 12

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = preprocess(X_train), preprocess(X_test)
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
_, img_rows, img_cols, channel = X_train.shape

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], channel, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], channel, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channel)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channel)

model = load_model('model_compressed.h5', custom_objects)
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
        C.EarlyStopping(patience=20),
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
