from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import scipy.stats

def plot_distribution(ax, value,
                      title="Distribution",
                      subtitle=None,
                      xlabel='Value',
                      ylabel='Frequency'):
    sns.distplot(value, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if subtitle is not None:
        title = title + "\n" + subtitle
    ax.set_title(title)

def generate_color_cycle(labels):
    unique_labels = np.unique(labels)
    unique_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, unique_colors))
    return [label_to_color[label] for label in labels], label_to_color

def plot_influence_correlation(ax,
                               x,
                               y,
                               label=None,
                               alpha=0.5,
                               size=5,
                               colors=None,
                               title="Predicted against actual influence",
                               subtitle=None,
                               xlabel="Actual influence",
                               ylabel="Predicted influence",
                               balanced=False,
                               equal=True,
                               spearmanr=True):
    # Compute data bounds
    minX, maxX = np.min(x), np.max(x)
    minY, maxY = np.min(y), np.max(y)

    if equal:
        minX = minY = min(minX, minY)
        maxX = maxY = max(maxX, maxY)

    if balanced:
        maxX = max(np.abs(minX), np.abs(maxX))
        minX = -maxX
        maxY = max(np.abs(minY), np.abs(maxY))
        minY = -maxY

    # Expand bounds
    padX = 0.05 * (maxX - minX)
    padY = 0.05 * (maxY - minY)
    minW, maxW = min(minX, minY), max(maxX, maxY)
    padW = max(padX, padY)

    # Plot x=y
    ax.plot([minW - padW, maxW + padW],
            [minW - padW, maxW + padW], color='grey', alpha=0.3)

    # Color groups of points if tagged
    if colors is None and label is not None:
        colors, label_to_color = generate_color_cycle(label)
        legend_elements = [ Line2D([0], [0], linewidth=0, marker='o',
                                   color=label_color, label=label_name, markersize=5)
                            for label_name, label_color in label_to_color.items() ]
        ax.legend(handles=legend_elements,
                  loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    # Randomize plot order for colors to show up better
    rng = np.random.RandomState(0)
    order = np.arange(len(x))
    rng.shuffle(order)
    x = x[order]
    y = y[order]
    assert len(x) == len(order)
    if colors is not None:
        assert len(x) == len(colors)
        colors = np.array(colors)[order]

    # Plot points
    ax.scatter(x, y, color=colors, alpha=alpha, s=size)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlim([minX - padX, maxX + padX])
    ax.set_ylim([minY - padY, maxY + padY])

    if subtitle is not None:
        title = title + "\n" + subtitle
    if spearmanr:
        rho, pval = scipy.stats.spearmanr(x, y)
        title = title + "\n" + "Spearman $\\rho$ = {}, $p$ = {}".format(rho, pval)
    ax.set_title(title)

def plot_against_subset_size(ax,
                             subset_tags,
                             subset_indices,
                             value,
                             title='Group self-influence against subset size',
                             xlabel='Group size',
                             ylabel='Influence',
                             subtitle=None):
    subset_sizes = np.array([len(indices) for indices in subset_indices])
    maxS = np.max(subset_sizes)
    maxV = np.max(value)

    label = subset_tags
    colors, label_to_color = generate_color_cycle(label)

    for label, label_color in label_to_color.items():
        cur_subsets = np.array(subset_tags) == label
        cur_sizes = subset_sizes[cur_subsets]
        cur_values = np.array(value)[cur_subsets]
        sort_idx = np.argsort(cur_sizes)
        cur_sizes = cur_sizes[sort_idx]
        cur_values = cur_values[sort_idx]
        ax.plot(cur_sizes, cur_values, c=label_color, label=label,
                alpha=0.5, marker='o', markersize=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    if subtitle is not None:
        title = title + "\n" + subtitle
    ax.set_title(title)
