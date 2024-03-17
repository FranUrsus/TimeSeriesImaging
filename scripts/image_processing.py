import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
import matplotlib.image as img
from PIL import Image


# return a transformed Gram Angular Field image (difference)
def transform_image_with_gram_angular_field(time_series, method='difference'):
    np_array_from_ts = np.array([time_series])
    gaf = GramianAngularField(method=method)
    gaf_ts = gaf.fit_transform(np_array_from_ts)
    return gaf_ts[0]


def transform_image_with_markov(time_series):
    np_array_from_ts = np.array([time_series])
    mtf = MarkovTransitionField(n_bins=8)
    mtf_ts = mtf.fit_transform(np_array_from_ts)
    return mtf_ts[0]


#

# plot a list of gram images displayed in n_rows * n_cols matrix format
def plot_all_gram_week_ts_images(gram_images, n_rows, n_cols):
    from mpl_toolkits.axes_grid1 import ImageGrid

    # Plot the n_rows*n_cols Gram angular fields for the week time series
    fig = plt.figure(figsize=(10, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1, share_all=True,
                     cbar_mode='single')
    for i, ax in enumerate(grid):
        im = ax.imshow(gram_images[i], cmap='rainbow', origin='lower')
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
    r = (np.arange(0, np.shape(data)[0]) / np.shape(data)[0]) + 0.1
    return phi, r


def plot_polar_coords(radious, tetha):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(tetha, radious)
    ax.set_rmax(1)
    ax.set_rticks(np.linspace(0, 1, 4))
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    ax.set_title("Time series polar coordinates representation", va='bottom')
    plt.show()


# Create a RGB Gram image from summ and dif images. ***Not used***
# Another alternative to generate RGB Gram images from time series in Gram matrices domain
def create_rgb_gram_image(gram_sum, gram_diff):
    x_train_gaf = np.concatenate((gram_sum, gram_diff, np.zeros(gram_diff.shape)), axis=-1)
    return x_train_gaf


# Save three channels (gram summ, diff and markov in corresponding folder)
def save_three_channels_images(input_nn_diff, input_nn_sum, input_nn_markov, labels):
    row_id = 0
    for diff, sum_, mk in zip(input_nn_diff, input_nn_sum, input_nn_markov):
        img.imsave(f'../data/images/input/next_day_cluster_{labels[row_id]}/mk_{row_id}.png', mk,cmap="rainbow")
        img.imsave(f'../data/images/input/next_day_cluster_{labels[row_id]}/gram_df_{row_id}.png', diff,cmap="rainbow")
        img.imsave(f'../data/images/input/next_day_cluster_{labels[row_id]}/gram_sum_{row_id}.png', sum_,cmap="rainbow")
        row_id += 1


# Save each original 1D time serie from dataset as png image
def save_rgb_images(time_series, labels, width=64, height=64):
    row_id = 0
    base_dir = '../data/images/1D/next_day_cluster_'
    for ts in time_series:
        plt.figure()
        plt.rcParams['figure.dpi'] = 64
        plt.plot(ts)
        plt.axis('off')
        plt.savefig(f'base_dir{labels[row_id]}/{row_id}.png', bbox_inches='tight', dpi='figure')
        plt.close()
        image = Image.open(f'base_dir{labels[row_id]}/{row_id}.png')
        new_image = image.resize((width, height))
        new_image.save(f'base_dir{labels[row_id]}/{row_id}.png')
        row_id += 1


def visualize_train_dataset(train_ds, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


def plot_rgb_channels(img):
    np_image = np.asarray(img)

    plt.figure(figsize=(12, 6))
    plt.subplot(131)

    plt.imshow(np_image[:, :, 0], cmap='Reds', vmin=0, vmax=255)
    plt.title("Red Channel")
    plt.subplot(132)

    plt.imshow(np_image[:, :, 1], cmap='Greens', vmin=0, vmax=255)
    plt.title("Green Channel")
    plt.subplot(133)

    plt.imshow(np_image[:, :, 2], cmap='Blues', vmin=0, vmax=255)
    plt.title("Blue Channel")

    plt.show()


## only uses for cube plots ad-hoc generation for graphical abstract and neural network backbone modeling
def cube_plot(axes, colors):
    # Create axis
    # axes = [3, 9, 3]

    # Create Data
    data = np.ones(axes, dtype=np.bool_)

    # Control Transparency
    alpha = 0.9

    # Control colour
    # colors = np.empty(axes + [4], dtype=np.float32)

    # colors[:] = [0, 1, 0, alpha]  # red

    # colors[0] = [1, 0, 0, alpha]  # red
    # colors[1] = [0, 1, 0, alpha]  # green
    # colors[2] = [0, 0, 1, alpha]  # blue

    # Plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Voxels is used to customizations of the
    # sizes, positions and colors.
    ax.voxels(data, facecolors=colors, edgecolors='k')


# return number of channer of a image
def get_nummber_of_channels(img_url):
    img_ = Image.open(img_url)
    img_np = np.asarray(img_)
    print("img_np.shape: ", img_np.shape)
    return img_np.shape


def plot_markov(time_series, markov, type_ts='week'):
    # Plot the time series and its Markov transition field
    width_ratios = (2, 7, 0.4)
    height_ratios = (2, 7)
    width = 6
    height = width * sum(height_ratios) / sum(width_ratios)
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 3, width_ratios=width_ratios,
                          height_ratios=height_ratios,
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    time_points = [*range(0, len(time_series), 1)]

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
    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.plot(time_points, time_series)
    ax_top.set_xticks(time_ticks)
    ax_top.set_xticklabels(time_tick_labels)
    ax_top.set_yticks(value_ticks)
    ax_top.set_yticklabels(value_ticks)
    ax_top.xaxis.tick_top()
    ax_top.set_xlim((0, len(time_series)))
    ax_top.set_yticklabels(value_ticks)

    # Plot the Gramian angular fields on the bottom right
    ax_mtf = fig.add_subplot(gs[1, 1])
    tuple_extent = (0, len(time_points), 0, len(time_points))
    im = ax_mtf.imshow(markov, cmap='rainbow', origin='lower', vmin=0., vmax=1.,
                       extent=tuple_extent)
    ax_mtf.set_xticks([])
    ax_mtf.set_yticks([])
    ax_mtf.set_title('Markov Transition Field', y=-0.09)

    # Add colorbar
    ax_cbar = fig.add_subplot(gs[1, 2])
    fig.colorbar(im, cax=ax_cbar)

    plt.show()


# Plot  time series in Gram Angular Field domain. type 'week' or 'cluster'
def plot_gram_angular_field(time_series, gaf_sm_ts, gaf_diff_ts, type_ts='week'):
    # Plot the time series and its recurrence plot
    width_ratios = (2, 7, 7, 0.4)
    height_ratios = (2, 7)
    width = 10
    height = width * sum(height_ratios) / sum(width_ratios)
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 4, width_ratios=width_ratios,
                          height_ratios=height_ratios,
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.1, hspace=0.1)

    # Define the ticks and their labels for both axes
    time_points = [*range(0, len(time_series), 1)]

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
    ax_top2 = fig.add_subplot(gs[0, 2])
    for ax in (ax_top1, ax_top2):
        ax.plot(time_points, time_series)
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_tick_labels)
        ax.set_yticks(value_ticks)
        ax.xaxis.tick_top()
        ax.set_xlim((0, len(time_series)))
    ax_top1.set_yticklabels(value_ticks)
    ax_top2.set_yticklabels([])

    # Plot the Gramian angular fields on the bottom right
    ax_gasf = fig.add_subplot(gs[1, 1])
    tuple_extent = (0, len(time_points), 0, len(time_points) )
    ax_gasf.imshow(gaf_sm_ts, cmap='rainbow', origin='lower',
                   extent=tuple_extent)
    ax_gasf.set_xticks([])
    ax_gasf.set_yticks([])
    ax_gasf.set_title('Gramian Angular Summation Field', y=-0.09)

    ax_gadf = fig.add_subplot(gs[1, 2])
    im = ax_gadf.imshow(gaf_diff_ts, cmap='rainbow', origin='lower',
                        extent=tuple_extent)
    ax_gadf.set_xticks([])
    ax_gadf.set_yticks([])
    ax_gadf.set_title('Gramian Angular Difference Field', y=-0.09)

    # Add colorbar
    ax_cbar = fig.add_subplot(gs[1, 3])
    fig.colorbar(im, cax=ax_cbar)

    plt.show()
