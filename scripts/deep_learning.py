from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import Adam
import keras
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf

import keras_tuner as kt

from sklearn import preprocessing

from keras.layers import *
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50


class DeepLearning:

    def __init__(self,
                 img_width,
                 img_height,
                 img_channels,
                 num_classes,
                 epochs=100,
                 ):
        # images
        self.input_shape = (img_width, img_height, img_channels)
        self.img_width = img_width
        self.img_height = img_height

        # dataset validation and train
        self.val_ds = None
        self.train_ds = None

        # classes
        self.num_classes = num_classes
        self.class_names = None

        # model
        self.model = None

        # history of model training and validation result
        self.history = None

        # epoch for models training
        self.epochs = epochs
        self.best_epoch = None

        # early stop
        self.early_stopping = None

        # tuner
        self.tuner = None

    # split dataset in train and test for supervised learning process
    # is valid only is images is classified on folder class structure
    def split_train_and_test(self, dataset_url, batch_size=32):
        data_dir = pathlib.Path(dataset_url)

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.1,
            subset="training",
            seed=123,
            labels='inferred',
            label_mode="categorical",
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.1,
            subset="validation",
            label_mode="categorical",
            labels='inferred',
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size)

        self.class_names = self.train_ds.class_names
        self.num_classes = len(self.class_names)

    # to avoid model over-training
    def get_early_stopping(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
        self.early_stopping = early_stopping
        return early_stopping

    def get_tuner(self, hp):
        tuner = kt.Hyperband(self.model_builder,
                             objective='val_accuracy',
                             max_epochs=10,
                             factor=3,
                             directory='../data/models/results',
                             project_name='kt_consumption_forecasting')
        self.tuner = tuner
        return tuner

    def show_results(self, h):
        acc = h.history['accuracy']
        val_acc = h.history['val_accuracy']

        loss = h.history['loss']
        val_loss = h.history['val_loss']

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

        # best epoch
        val_acc_per_epoch = h.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))
        self.best_epoch = best_epoch

    # build deep learning model generator based on hyperparameter tuning
    def model_builder(self, hp):
        # create deep learning sequential model
        model = keras.Sequential()

        # set image input shape for deep learning process
        model.add(Input(shape=self.input_shape))

        # rescaling images pixel values on channels in [0,1] range
        #model.add(Rescaling(1. / 255.)),

        # First conv-pooling
        model.add(Conv2D(hp.Int("conv_1", min_value=32, max_value=64, step=32),
                         kernel_size=3,
                         activation='softmax'))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second conv-pooling
        model.add(Conv2D(hp.Int("conv_2", min_value=16, max_value=32, step=16),
                         kernel_size=3,
                         activation='softmax'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # fully connected layer
        model.add(Flatten())

        model.add(Dense(self.num_classes, activation='softmax'))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001

        lr = hp.Choice("learning_rate",
                       values=[1e-1, 1e-2, 1e-3])
        opt = Adam(lr)

        model.compile(optimizer=opt,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        self.model = model

        return model
