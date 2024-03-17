import matplotlib.pyplot as plt
import random
import os
from sklearn.model_selection import train_test_split
import numpy as np


# return a generated random color
def random_color_generator():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r, g, b


# plot a time_series with week separator
def plot_time_series_with_week_separator(time_series):
    plt.figure(figsize=(20, 10))
    time_series.plot(marker='o', markersize=3)
    plt.xlabel('Hour', fontsize=15)
    plt.ylabel('Consumption (Wh)', fontsize=15)
    week_separator_vertical_line_coords = [*range(23, 169, 24)]
    i = 0
    colors = ['blue', 'green', 'red', 'cyan', 'pink', 'yellow', 'orange']
    for weekCoords in week_separator_vertical_line_coords:
        plt.vlines(x=weekCoords, colors=colors[i], ls='--', lw=2, label='day ' + str(i + 1), ymin=0,
                   ymax=max(time_series))
        i += 1
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', title="Day of the week")
    plt.show()


# plot a time_series
def plot_time_series(time_series):
    time_series.plot()
    plt.xlabel('Hour', fontsize=15)
    plt.ylabel('Consumption (Wh)', fontsize=15)
    plt.show()


def create_folder_structure(next_day_clusters):
    path = "../data/images/input/"
    for clus in next_day_clusters:
        path_clus = path + 'next_day_cluster_' + str(clus)
        if not (os.path.exists(path_clus)):
            os.mkdir(path_clus)


## create folder sctucture over 1D folder to save original images clasified by cluster next day
def create_folder_structure_origin_images(next_day_clusters):
    path = "../data/images/1D/"
    for clus in next_day_clusters:
        path_clus = path + 'next_day_cluster_' + str(clus)
        if not (os.path.exists(path_clus)):
            os.mkdir(path_clus)


# return list with class names
def get_cluster_class_names(next_day_clusters):
    classes = []
    for clus in next_day_clusters:
        path_clus = 'next_day_cluster_' + str(clus)
        classes.append(path_clus)
    return classes


def prepare_data_for_lstm(df):
    transposed_df = df.T
    df_unestack = transposed_df.unstack().reset_index(drop=True).T
    return df_unestack


# plot predicted consumption hourly day with lstm vs real consumption day
def plot_predicted_over_real(n_iter, prediction_no_scaled, y_test):
    for i in range(n_iter):
        plt.figure()
        plt.plot(prediction_no_scaled[i], color='red', label='predicted')
        plt.plot(y_test[i], color='green', label='real')
        plt.legend()
        plt.show()


# Split data in validation and train sets
def split_data(x, target):
    x_train, x_valid, y_train, y_valid = train_test_split(x, target, test_size=0.25, random_state=36)
    return x_train, x_valid, y_train, y_valid


def create_three_channel_dataset(df,
                                 g_dif_channel,
                                 g_sum_channel,
                                 g_markov_channel):
    g_df_images = np.zeros((len(df), 168, 168))
    g_sum_images = np.zeros((len(df), 168, 168))
    mk_images = np.zeros((len(df), 168, 168))
    counter = 0
    for g_diff, g_summ, mk in zip(g_dif_channel, g_sum_channel, g_markov_channel):
        g_df_images[counter] = g_diff
        g_sum_images[counter] = g_summ
        mk_images[counter] = mk
        counter = counter + 1
    return g_df_images, g_sum_images, mk_images
