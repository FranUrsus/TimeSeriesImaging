import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from pyts.image import GramianAngularField
import matplotlib.image as img


# return a transformed Gram Angular Field image (difference)
def transform_image_with_gram_angular_field(time_series, method='difference'):
    np_array_from_ts = np.array([time_series])
    gaf = GramianAngularField(method=method)
    gaf_ts = gaf.fit_transform(np_array_from_ts)
    return gaf_ts[0]


# return a transformed Gram Angular Field image (sum)
def transform_image_with_gram_angular_field_sum(time_series, method='summation'):
    np_array_from_ts = np.array([time_series])
    gaf = GramianAngularField(method=method)
    gaf_ts = gaf.fit_transform(np_array_from_ts)
    return gaf_ts[0]


# Plot  time series in Gram Angular Field domain
# type 'week' or 'cluster'

def plot_gram_angular_field(time_series, gaf_ts_diff, type_ts='week'):
    # Plot the time series and its recurrence plot
    width_ratios = (2, 7, 0.4)
    height_ratios = (2, 7)
    width = 10
    height = width * sum(height_ratios) / sum(width_ratios)
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 3, width_ratios=width_ratios,
                          height_ratios=height_ratios,
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.1, hspace=0.1)
    time_points = [*range(0, len(time_series), 1)]
    # Define the ticks and their labels for both axes

    if type_ts == 'week':
        time_ticks = np.linspace(0, len(time_points), 8)
        time_tick_labels = time_ticks
    else:
        time_ticks = np.linspace(0, 24, 5)
        time_tick_labels = time_ticks

    value_ticks = [*range(0, max(time_series), 500)]
    reversed_value_ticks = value_ticks[::-1]

    # Plot the time series on the left with inverted axes
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.plot(time_series, time_points)
    ax_left.set_xticks(reversed_value_ticks)
    ax_left.set_xticklabels(reversed_value_ticks, rotation=90)
    ax_left.set_yticks(time_ticks)
    ax_left.set_yticklabels(time_tick_labels, rotation=90)
    ax_left.set_ylim((0, max(time_points)))
    ax_left.invert_xaxis()

    # Plot the time series on the top
    ax_top1 = fig.add_subplot(gs[0, 1])
    ax_top1.plot(time_points, time_series)
    ax_top1.set_xticks(time_ticks)
    ax_top1.set_xticklabels(time_tick_labels)
    ax_top1.set_yticks(value_ticks)
    ax_top1.xaxis.tick_top()
    ax_top1.set_xlim((0, len(time_series)))
    ax_top1.set_yticklabels(value_ticks)

    # Plot the Gram Angular Fields on the bottom right

    tuple_extent = (0, max(time_points), 0, max(time_points),)
    ax_gaf_diff = fig.add_subplot(gs[1, 1])
    im = ax_gaf_diff.imshow(gaf_ts_diff, cmap='rainbow', origin='lower',
                            extent=tuple_extent)
    ax_gaf_diff.set_xticks([])
    ax_gaf_diff.set_yticks([])
    ax_gaf_diff.set_title('Gram Angular Difference Field', y=-0.09)

    # Add color_bar
    ax_cbar = fig.add_subplot(gs[1, 2])
    fig.colorbar(im, cax=ax_cbar)

    plt.show()


# plot a list of gram images displayed in n_rows * n_cols matrix format
def plot_all_gram_week_ts_images(gram_images, n_rows, n_cols):
    from mpl_toolkits.axes_grid1 import ImageGrid

    # Plot the n_rows*n_cols Gram angular fields for the week time series
    fig = plt.figure(figsize=(10, 5))

    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1, share_all=True,
                     cbar_mode='single')
    for i, ax in enumerate(grid):
        im = ax.imshow(gram_images[i], cmap='rainbow', origin='lower', vmin=-1., vmax=1.)
        grid[0].get_yaxis().set_ticks([])
        grid[0].get_xaxis().set_ticks([])
        plt.colorbar(im, cax=grid.cbar_axes[0])
        ax.cax.toggle_label(True)

    fig.suptitle(f"Gramian angular difference fields for the {n_rows * n_cols} first week consumption time series", y=1)

    plt.show()


# input datatype data : normalize time serie in range [0,1]
def normalize_ts(ts):
    return (ts - min(ts)) / (max(ts) - min(ts))


# return polar coords of time series representation
# phi: angles
# r: radious
def polar_rep(data):
    phi = np.arccos(data)
    r = data
    return phi, r


def plot_polar_coords(tetha, radious):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(tetha, radious)
    ax.set_rmax(1)
    ax.set_rticks(np.linspace(0, 1, 3))
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    ax.set_title("Time series polar coordinates representation", va='bottom')
    plt.show()


# Create a RGB Gram image from summ and dif images. ***Not used***
# Another alternative to generate RGB Gram images from time series in Gram matrices domain
def create_rgb_gramian_image(gram_sum, gram_diff):
    x_train_gaf = np.concatenate((gram_sum, gram_diff, np.zeros(gram_diff.shape)), axis=-1)
    return x_train_gaf


# Save time serie in gramian matrix format as png
def save_rgb_images(gram_images_df):
    import matplotlib.image as img
    row_id = 0
    for ts_gram_matrix in gram_images_df:
        img.imsave(f'../data/images/input/{row_id}.png', ts_gram_matrix, cmap="rainbow")
        row_id += 1
