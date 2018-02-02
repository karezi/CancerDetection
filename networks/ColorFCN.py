import keras
from keras.layers import Conv2D, MaxPool2D
from keras.models import Sequential
from keras.utils import plot_model


class ColorFCN(object):

    _model = None

    def __init__(self, display=None, draw_framework=False):
        self._construct_model(display)
        if draw_framework:
            self.draw_framework()

    def get_model(self):
        return self._model

    def _construct_model(self, display=None):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(None, None, 3)))
        model.add(MaxPool2D())

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D())

        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D())

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Conv2D(128, (3, 3), padding='same'))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Conv2D(32, (3, 3), padding='same'))

        model.add(keras.layers.Conv2D(1, (3, 3), padding='same'))
        model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
        if display:
            display(model)
        else:
            pass
        self._model = model

    def draw_framework(self):
        plot_model(self._model, to_file='ColorFCN.png', show_shapes=True, show_layer_names=True)


if __name__ == '__main__':
    ColorFCN(draw_framework=True)
