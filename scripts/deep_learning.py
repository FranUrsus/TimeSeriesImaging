from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import keras


class DeepLearning:
    def __init__(self, num_images, num_classes, img_width, img_height, img_channels, model=Sequential()):
        self.input_shape = (num_images, img_width, img_height, img_channels)
        self.num_classes = num_classes
        self.model = model


def train_model(self):
    input_shape = self.input_shape

    self.model.add(Conv2D(32,
                          kernel_size=(3, 3),
                          activation='relu',
                          input_shape=input_shape))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.25))

    self.model.add(Conv2D(32,
                          kernel_size=(3, 3),
                          activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.25))

    self.model.add(Flatten())

    self.model.add(Dense(128, activation='relu'))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(self.num_classes, activation='softmax'))

    self.model.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=keras.optimizers.Adam(),
                       metrics=['accuracy'])

    self.model.fit(x_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_data=(x_test, y_test))

    return self.model
