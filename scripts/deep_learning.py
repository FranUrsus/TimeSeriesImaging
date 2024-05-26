from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import Adam
import tensorflow.keras
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf


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
                 class_names="names",
                 epochs=10,
                 batch_size=32
                 ):
        # images
        self.data_augmentation = None
        self.batch_size = batch_size
        self.test_ds = None
        self.class_names = class_names
        self.input_shape = (img_width, img_height, img_channels)
        self.img_width = img_width
        self.img_height = img_height

        self.img_channels = img_channels

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

    # Split dataset in train and test for supervised learning process
    # is valid only is images is classified on folder class structure
    def split_train_and_test(self, dataset_url, batch_size=32):
        data_dir = pathlib.Path(dataset_url)

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.3,
            subset="training",
            seed=123,
            color_mode='rgb',
            label_mode="categorical",
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.3,
            subset="validation",
            label_mode="categorical",
            color_mode='rgb',
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size)

        self.class_names = self.train_ds.class_names
        self.num_classes = len(self.class_names)

    def get_early_stopping(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                          patience=3)
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

    def init_data_augmentation(self):
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal", input_shape=(self.img_height, self.img_width, 3)),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(factor=(0.8, 1.4))
            ]
        )
        self.data_augmentation = data_augmentation

    def random_augmented_images(self):
        class_names = self.train_ds.class_names
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_ds.take(1):
            for i in range(9):
                augmented_images = self.data_augmentation(images)
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(augmented_images[i].numpy().astype("uint8"))
                # plt.title(class_names[labels[i]])
                plt.axis("off")

    def random_augmented_images_m3(self,images):
        class_names = self.train_ds.class_names
        plt.figure(figsize=(10, 10))
        for image in images:
            for i in range(9):
                augmented_images = self.data_augmentation(image)
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(augmented_images[i].numpy().astype("uint8"))
                # plt.title(class_names[labels[i]])
                plt.axis("off")