from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


def preprocess(X):
    return X.astype('float32') / 255 * 2 - 1


class_num = 10
batch_size = 128
epochs = 300

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = preprocess(X_train), preprocess(X_test)
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
_, img_rows, img_cols, channels = X_train.shape

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], channels, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)


def gen_model():
    # refer to http://torch.ch/blog/2015/07/30/cifar.html
    img_input = Input(shape=(img_rows, img_cols, channels))

    h = img_input

    h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
    h = BatchNormalization()(h)
    h = Dropout(0.3)(h)
    h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
    h = BatchNormalization()(h)
    h = MaxPool2D((2, 2), strides=(2, 2))(h)

    h = Conv2D(128, (3, 3), activation='relu', padding='same')(h)
    h = BatchNormalization()(h)
    h = Dropout(0.4)(h)
    h = Conv2D(128, (3, 3), activation='relu', padding='same')(h)
    h = BatchNormalization()(h)
    h = MaxPool2D((2, 2), strides=(2, 2))(h)

    h = Conv2D(256, (3, 3), activation='relu', padding='same')(h)
    h = BatchNormalization()(h)
    h = Dropout(0.4)(h)
    h = Conv2D(256, (3, 3), activation='relu', padding='same')(h)
    h = BatchNormalization()(h)
    h = Dropout(0.4)(h)
    h = Conv2D(256, (3, 3), activation='relu', padding='same')(h)
    h = BatchNormalization()(h)
    h = MaxPool2D((2, 2), strides=(2, 2))(h)

    h = Conv2D(512, (3, 3), activation='relu', padding='same')(h)
    h = BatchNormalization()(h)
    h = Dropout(0.4)(h)
    h = Conv2D(512, (3, 3), activation='relu', padding='same')(h)
    h = BatchNormalization()(h)
    h = Dropout(0.4)(h)
    h = Conv2D(512, (3, 3), activation='relu', padding='same')(h)
    h = BatchNormalization()(h)
    h = MaxPool2D((2, 2), strides=(2, 2))(h)

    h = Conv2D(512, (3, 3), activation='relu', padding='same')(h)
    h = BatchNormalization()(h)
    h = Dropout(0.4)(h)
    h = Conv2D(512, (3, 3), activation='relu', padding='same')(h)
    h = BatchNormalization()(h)
    h = Dropout(0.4)(h)
    h = Conv2D(512, (3, 3), activation='relu', padding='same')(h)
    h = BatchNormalization()(h)
    h = MaxPool2D((2, 2), strides=(2, 2))(h)

    h = Flatten()(h)

    h = Dropout(0.5)(h)
    h = Dense(512, activation='relu')(h)
    h = BatchNormalization()(h)

    h = Dropout(0.5)(h)
    h = Dense(class_num, activation='softmax')(h)

    model = Model(img_input, h)
    return model


model = gen_model()
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'],
)

datagen = ImageDataGenerator(
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

datagen.fit(X_train)

model.fit_generator(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=X_train.shape[0] // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[
        EarlyStopping(patience=20),
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
model.save('model_raw.h5')
