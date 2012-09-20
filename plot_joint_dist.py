#!/usr/bin/env python
r"""plot 2D histograms along combinations of variables
ERS Aug 2012
"""
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import copy
from scipy import optimize
import itertools
import scipy.ndimage as ndimage
from optparse import OptionParser
# TODO: make binning, conf contours come from command line
# TODO: marginal axis looks off-by-one
# TODO: variable confidence contour color
# TODO: let histogram return the bin axis


def conf_level(histogram, clevels):
    r"""given a histogram, find the value within which clevels percent of the
    probability is enclosed"""
    norm_histo = copy.deepcopy(histogram)
    norm = np.max(histogram)
    norm_histo /= norm
    total = np.sum(norm_histo)
    clevel = np.nan  # sanity

    def frac_within(threshold):
        frac = np.sum(norm_histo[np.where(norm_histo > threshold)]) / total
        return frac - clevel

    retval = []
    for clevel in clevels:
        retval.append(optimize.bisect(frac_within, 0., 1.) * norm)

    return retval


def find_range(data_in, nsigma=3.):
    r"""helper to find ranges of plots"""
    # TODO: make this more robust to outliers
    center = np.mean(data_in)
    delta = np.std(data_in)

    sigmaleft = center - nsigma * delta
    sigmaright = center + nsigma * delta

    # let the plot range only extend to where data exist
    # leaving a 30% gap
    gap = 0.3 * delta
    left = max([sigmaleft, min(data_in) - gap])
    right = min([sigmaright, max(data_in) + gap])

    return [left, right]


def rescale_log(data_range, min_thresh=-2., max_thresh=3.):
    r"""rescale data for scientific notation"""
    largest = np.max(np.log10(np.abs(np.array(data_range))))
    if largest > min_thresh and largest < max_thresh:
        return None
    else:
        return int(largest)


def plot_2d_histo(x_data, y_data, nsigma=3.,
                  x_label=None, y_label=None, bins=25, mbins=50,
                  conf_contours=[0.5, 0.95],
                  conf_labels=["$50$", "$95$"],
                  plot_filename="plot_2d_histo.eps",
                  file_format="eps"):
    r"""Make the 2d histogram and plot it
    `x_data` and `y_data` are the data to bin
    `x_range` is the range list [xmin, xmax]
    `y_range` is the range list [ymin, ymax]
    `x_label` is the string that appears as the x label; LaTeX rendered?
    `y_label` is the string that appears as the x label; LaTeX rendered?
    `bins` is the number of bins to use along each axis
    `mbins` is the number of bins to use along the marginalized axes

    ls="steps" also looks good in the marginalized dist
    TODO: this currently rescales x,y_data (unsafe)
    """
    x_range = find_range(x_data, nsigma=nsigma)
    rescale_val = rescale_log(x_range)
    if rescale_val:
        print "rescaling %s by %d" % (x_label, rescale_val)
        x_data /= 10. ** rescale_val
        x_label += "\\cdot 10^{%d}" % -rescale_val

        x_range = find_range(x_data, nsigma=nsigma)

    y_range = find_range(y_data, nsigma=nsigma)
    rescale_val = rescale_log(y_range)
    if rescale_val:
        print "rescaling %s by %d" % (y_label, rescale_val)
        y_data /= 10. ** rescale_val
        y_label += "\\cdot 10^{%d}" % -rescale_val

        y_range = find_range(y_data, nsigma=nsigma)

    fig = plt.figure(1, figsize=(7, 7))

    fig.subplots_adjust(hspace=0.001, wspace=0.001, left=0.15, bottom=0.12,
                        top=0.975, right=0.98)

    subplot_grid = gridspec.GridSpec(2, 2, width_ratios=[1, 4],
                                     height_ratios=[4, 1])

    plt.subplot(subplot_grid[1])

    # if no range is given, use the full range
    if x_range is None:
        x_range = (np.min(x_data), np.max(x_data))

    if y_range is None:
        y_range = (np.min(y_data), np.max(y_data))

    # first bin the data and plot it as a colormesh
    histo_2d, xedges, yedges = np.histogram2d(x_data, y_data,
                                              bins=[bins, bins],
                                              range=[x_range, y_range],
                                              normed=False)

    histo_2d = np.transpose(histo_2d)
    # gray, binary and jet are good options
    plt.pcolormesh(xedges, yedges, histo_2d, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])

    clevels = conf_level(histo_2d, conf_contours)
    #print clevels
    linestyles = []
    colors = []
    fmtdict = {}
    for (clevel, conf_label) in zip(clevels, conf_labels):
        linestyles.append('--')
        colors.append('blue')
        fmtdict[clevel] = conf_label

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    histo_2d_smooth = ndimage.gaussian_filter(histo_2d, sigma=1., order=0)
    contours = plt.contour(histo_2d_smooth, extent=extent, levels=clevels,
                     linestyles=linestyles, colors=colors, linewidths=2)

    plt.clabel(contours, fmt=fmtdict, inline=True, fontsize=20)

    plt.xlim(x_range)
    plt.ylim(y_range)

    (histo_x, xedges) = np.histogram(x_data, bins=mbins,
                                      range=x_range, normed=True)

    (histo_y, yedges) = np.histogram(y_data, bins=mbins,
                                      range=y_range, normed=True)

    x_vec = 0.5 * (xedges[1:] + xedges[:-1])
    y_vec = 0.5 * (yedges[1:] + yedges[:-1])

    # interpret ~ as a divider between variable name and units
    x_label_sp = "$%s$" % "$ $".join(x_label.split("~"))
    y_label_sp = "$%s$" % "$ $".join(y_label.split("~"))

    # plot the marginalized x data
    plt.subplot(subplot_grid[3])
    plt.plot(x_vec, histo_x, '-', lw=3, color='black')
    #plt.ticklabel_format(style="sci", axis='x', scilimits=(1, 3))
    plt.xticks(fontsize=16)
    plt.yticks([])
    plt.xlabel(x_label_sp, fontsize=24)
    #plt.ylabel(r'$\cal L$', fontsize=24)
    plt.xlim(x_range)
    plt.ylim(0.0, 1.1 * np.max(histo_x))

    # plot the marginalized y data
    plt.subplot(subplot_grid[0])
    plt.plot(histo_y, y_vec, '-', lw=3, color='black')
    #plt.ticklabel_format(style="sci", axis='y', scilimits=(1, 3))
    plt.yticks(fontsize=16)
    plt.xticks([])
    #plt.xlabel(r'$\cal L$', fontsize=24)
    plt.ylabel(y_label_sp, fontsize=24)
    plt.xlim(0.0, 1.1 * np.max(histo_y))
    plt.ylim(y_range)

    plt.savefig(plot_filename, format=file_format)
    #plt.show()


def plot_chains(filename, nsigma=3., separate=False,
                output=None, file_format="eps"):
    r"""Given an hd5 file written by wrap_emcee, plot the joint distribution
    of all the parameters in the output chains"""
    np.set_printoptions(threshold='nan')
    mcmc_data = h5py.File(filename, "r")
    chain_data = mcmc_data['chain']
    desc_data = mcmc_data['desc']
    print chain_data.keys(), "plot range (nsigma): ", nsigma
    basename = os.path.splitext(filename)[0]

    if not separate:
        if output is None:
            plot_filename = "%s.%s" % (basename, file_format)
        else:
            plot_filename = output

        plot_chains_triangle(chain_data, desc_data, plot_filename, nsigma=3.,
                             var_list=None, conf_contours=[0.5, 0.95],
                             conf_labels=["$50$", "$95$"],
                             size_multiplier=3., bins=50, mbins=50,
                             file_format=file_format)

    else:
        if output is None:
            root = "./"
        else:
            root = output

        for pair in itertools.combinations(chain_data, 2):
            plot_filename = "%s/%s_%s_with_%s.%s" % \
                            (root, basename, pair[0], pair[1], file_format)

            print plot_filename

            plot_2d_histo(copy.deepcopy(chain_data[pair[0]].value),
                          copy.deepcopy(chain_data[pair[1]].value),
                          x_label=copy.deepcopy(desc_data[pair[0]].value),
                          y_label=copy.deepcopy(desc_data[pair[1]].value),
                          bins=50, plot_filename=plot_filename,
                          file_format=file_format)

    mcmc_data.close()


def plot_chains_triangle(chain_data, desc_data, plot_filename, nsigma=3.,
                         var_list=None, conf_contours=[0.5, 0.95],
                         conf_labels=["$50$", "$95$"], file_format="eps",
                         size_multiplier=3., bins=50, mbins=50):
    r"""plot a triangle of joint variable distributions (J. Chluba, ERS)
    `var_list` is an optional list of variables specifying order/subsets
    `size_multiplier` * number of variables = size of plot
    """
    print "making a joint param triangle plot: ", plot_filename

    if var_list is None:
        # can not iterate over keys in a grid, so make a list
        var_list = chain_data.keys()

    nvars = len(var_list)
    print "number of vars = ", nvars

    plot_size = nvars * size_multiplier
    fig = plt.figure(1, figsize=(plot_size, plot_size))
    fig.subplots_adjust(hspace=0.02, wspace=0.02, left=0.02, bottom=0.05,
                        top=0.98, right=0.98)

    for x_ind in range(0, nvars):
        for y_ind in range(x_ind, nvars):
            print "plotting section: ", y_ind, x_ind
            plt.subplot2grid((nvars, nvars), (x_ind, y_ind))

            # if the plot is along the diagonal, plot marginalized
            if x_ind == y_ind:
                m_data = copy.deepcopy(chain_data[var_list[x_ind]].value)
                m_label = desc_data[var_list[x_ind]].value
                m_range = find_range(m_data, nsigma=nsigma)

                rescale_val = rescale_log(m_range)
                if rescale_val:
                    print "rescaling %d by %d" % (x_ind, rescale_val)
                    m_data /= 10. ** rescale_val
                    m_label += "\\cdot 10^{%d}" % -rescale_val

                m_label_sp = "$%s$" % "$ $".join(m_label.split("~"))
                print m_label_sp

                m_range = find_range(m_data, nsigma=nsigma)

                (m_histo, m_edges) = np.histogram(m_data, bins=mbins,
                                       range=m_range, normed=True)

                m_vec = 0.5 * (m_edges[1:] + m_edges[:-1])

                # can also add ls="steps"
                plt.plot(m_vec, m_histo, '-', lw=3, color='black')
                #plt.ticklabel_format(style="sci", axis='x', scilimits=(1, 3))
                plt.yticks([])
                plt.xticks(fontsize=10)
                plt.xlabel(m_label_sp, fontsize=20)
                plt.xlim(m_range)
                plt.ylim(0.0, 1.1 * np.max(m_histo))
            else:  # plot the joint distribution
                x_data = chain_data[var_list[x_ind]].value
                x_range = find_range(x_data, nsigma=nsigma)
                y_data = chain_data[var_list[y_ind]].value
                y_range = find_range(y_data, nsigma=nsigma)

                histo_2d, xedges, yedges = np.histogram2d(x_data, y_data,
                                                    bins=[bins, bins],
                                                    range=[x_range, y_range],
                                                    normed=False)

                # gray, binary and jet are good options
                plt.pcolormesh(xedges, yedges, histo_2d, cmap=plt.cm.binary)
                plt.xticks([])
                plt.yticks([])

                clevels = conf_level(histo_2d, conf_contours)
                linestyles = []
                colors = []
                fmtdict = {}
                for (clevel, conf_label) in zip(clevels, conf_labels):
                    linestyles.append('--')
                    colors.append('blue')
                    fmtdict[clevel] = conf_label

                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                histo_2d_smooth = ndimage.gaussian_filter(histo_2d, sigma=1.,
                                                          order=0)

                contours = plt.contour(histo_2d_smooth, extent=extent,
                                       levels=clevels, linestyles=linestyles,
                                       colors=colors, linewidths=2)

                plt.clabel(contours, fmt=fmtdict, inline=True, fontsize=14)

                plt.xlim(x_range)
                plt.ylim(y_range)

    plt.savefig(plot_filename, format=file_format)


def main():
    r"""parse arguments to plot_chains from cmd line"""
    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")

    parser.add_option("-n", "--nsigma",
                      action="store",
                      dest="nsigma",
                      default=3.,
                      help="Range (in sigma) of axes")

    parser.add_option("-o", "--output",
                      action="store",
                      dest="output",
                      default=None,
                      help="Filename or output root (for -s)")

    parser.add_option("-f", "--format",
                      action="store",
                      dest="file_format",
                      default="png",
                      help="File format")

    parser.add_option("-s", "--separate",
                      action="store_true",
                      dest="separate",
                      default=False,
                      help="make separate plots for each joint var")

    (options, args) = parser.parse_args()
    optdict = vars(options)

    if len(args) != 1:
        parser.error("wrong number of arguments")

    #print options, args
    plot_chains(args[0], **optdict)


if __name__ == '__main__':
    main()
