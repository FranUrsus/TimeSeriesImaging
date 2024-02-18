import pyts
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField

#return a transformed GramianAngularField image
def transformImageWithGramianAngularField(time_serie,method='summation'):

    X = np.array([time_serie])
    gaf = GramianAngularField(method=method)
    gaf_ts = gaf.fit_transform(X)
    return gaf_ts

#Plot transformated time serie to Gramian Angular Field Domain
#type week or cluster

def plotGramianAngularField (time_serie,gaf_ts_sum, gaf_ts_diff, type = 'week'):

    # Plot the time series and its recurrence plot
    width_ratios = (2, 7, 7, 0.4)
    height_ratios = (2, 7)
    width = 10
    height = width * sum(height_ratios) / sum(width_ratios)
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 4,  width_ratios=width_ratios,
                          height_ratios=height_ratios,
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.1, hspace=0.1)
    time_points = [*range(0, len(time_serie), 1)]
    # Define the ticks and their labels for both axes

    if type == 'week':
        time_ticks = np.linspace(0, len(time_points), 8)
        time_ticklabels = time_ticks 
    else:
        time_ticks = np.linspace(0, 24, 5)
        time_ticklabels = time_ticks
        
    value_ticks = [*range(0, max(time_serie), 500)] 
    reversed_value_ticks = value_ticks[::-1]
    
    # Plot the time series on the left with inverted axes
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.plot(time_serie, time_points)
    ax_left.set_xticks(reversed_value_ticks)
    ax_left.set_xticklabels(reversed_value_ticks, rotation=90)
    ax_left.set_yticks(time_ticks)
    ax_left.set_yticklabels(time_ticklabels, rotation=90)
    ax_left.set_ylim((0, max(time_points)))
    ax_left.invert_xaxis()
    
    
    # Plot the time series on the top
    ax_top1 = fig.add_subplot(gs[0, 1])
    ax_top2 = fig.add_subplot(gs[0, 2])
    for ax in (ax_top1, ax_top2):
        ax.plot(time_points, time_serie)
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_ticklabels)
        ax.set_yticks(value_ticks)
        ax.xaxis.tick_top()
        ax.set_xlim((0, len(time_serie)))
    ax_top1.set_yticklabels(value_ticks)
    ax_top2.set_yticklabels([])
    
    # Plot the Gramian angular fields on the bottom right
    ax_gaf_sum = fig.add_subplot(gs[1, 1])
    ax_gaf_sum.imshow(gaf_ts_sum[0], cmap='rainbow', origin='lower',
                   extent=[0, max(time_serie), 0, max(time_serie)])
    ax_gaf_sum.set_xticks([])
    ax_gaf_sum.set_yticks([])
    ax_gaf_sum.set_title('Gramian Angular Summation Field', y=-0.09)

    ax_gaf_diff = fig.add_subplot(gs[1, 2])
    im = ax_gaf_diff.imshow(gaf_ts_diff[0], cmap='rainbow', origin='lower',
                        extent=[0, 4 * np.pi, 0, 4 * np.pi])
    ax_gaf_diff.set_xticks([])
    ax_gaf_diff.set_yticks([])
    ax_gaf_diff.set_title('Gramian Angular Difference Field', y=-0.09)

    # Add colorbar
    ax_cbar = fig.add_subplot(gs[1, 3])
    fig.colorbar(im, cax=ax_cbar)
    
    plt.show()
   