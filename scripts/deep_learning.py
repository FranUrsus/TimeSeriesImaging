import tensorflow.data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Rescaling
import keras
import pathlib
import keras as tf
import matplotlib.pyplot as plt


class DeepLearning:

    def __init__(self,
                 img_width,
                 img_height,
                 img_channels,
                 batch_size=32,
                 epochs=100,
                 model=Sequential()):
        self.history = None
        self.batch_size = batch_size
        self.class_names = None
        self.val_ds = None
        self.train_ds = None
        self.input_shape = (img_width, img_height, img_channels)
        self.num_classes = None
        self.model = model
        self.epochs = epochs
        self.img_width = img_width
        self.img_height = img_height

    # split dataset in train and test for supervised learning process
    def split_train_and_test(self, dataset_url):
        data_dir = pathlib.Path(dataset_url)

        self.train_ds = tf.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.8,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.val_ds = tf.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        autotune = tensorflow.data.AUTOTUNE

        self.train_ds = self.train_ds.cache().prefetch(buffer_size=autotune)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=autotune)

        self.class_names = self.train_ds.class_names

    # train a deep learning model for next day consumption hourly forecasting
    def train_model(self):
        self.num_classes = len(self.class_names)

        self.model.add(Rescaling(1. / 255, input_shape=self.input_shape)),

        self.model.add(Conv2D(32,
                              kernel_size=(3, 3),
                              activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64,
                              kernel_size=(3, 3),
                              activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())

        self.model.add(Dense(64, activation='relu'))

        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])

        self.history = self.model.fit(train_data=self.train_ds,
                                      epochs=self.epochs,
                                      validation_data=self.val_ds)

    def show_results(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
