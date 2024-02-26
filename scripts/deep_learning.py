from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split


class DeepLearning:

    def __init__(self,
                 num_images,
                 num_classes,
                 img_width,
                 img_height,
                 img_channels,
                 batch_size=32,
                 epochs=100,
                 model=Sequential()):

        self.input_shape = (num_images, img_width, img_height, img_channels)
        self.num_classes = num_classes
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs


def split_train_and_test(self, dataset_url, labels):

    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(dataset_url, labels, test_size=0.33)


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

    self.model.fit(self.x_train, self.y_train,
                   batch_size=self.batch_size,
                   epochs=self.epochs,
                   verbose=1,
                   validation_data=(self.x_test, self.y_test))

    return self.model
