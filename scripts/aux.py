import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split


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


