#!/usr/bin/env python
r"""plot 2D histograms along combinations of variables"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import copy
from scipy import optimize
import itertools
import scipy.ndimage as ndimage
from optparse import OptionParser


def conf_level(histogram, clevels):
    norm_histo = copy.deepcopy(histogram)
    norm = np.max(histogram)
    norm_histo /= norm
    total = np.sum(norm_histo)

    def frac_within(threshold):
        frac = np.sum(norm_histo[np.where(norm_histo > threshold)]) / total
        return frac - clevel

    retval = []
    for clevel in clevels:
        retval.append(optimize.bisect(frac_within, 0., 1.) * norm)
    
    return retval


def plot_2d_histo(x_data, y_data, xrange=None, yrange=None,
                  xlabel=None, ylabel=None, bins=25, hbins=50,
                  conf_contours=[0.5, 0.95],
                  conf_labels=["$50$", "$95$"],
                  filename="plot_2d_histo.eps"):
    r"""Make the 2d histogram and plot it
    `x_data` and `y_data` are the data to bin
    `xrange` is the range list [xmin, xmax]
    `yrange` is the range list [ymin, ymax]
    `xlabel` is the string that appears as the x label; LaTeX rendered?
    `ylabel` is the string that appears as the x label; LaTeX rendered?
    `bins` is the number of bins to use along each axis
    `hbins` is the number of bins to use along the marginalized axes
    """
    # set up the plot geometry
    fig = plt.figure(1, figsize=(7,7))

    fig.subplots_adjust(hspace=0.001, wspace=0.001, left=0.12, bottom=0.12,
                        top=0.975, right=0.98)

    gs = gridspec.GridSpec(2, 2, width_ratios=[1,4], height_ratios=[4,1])

    plt.subplot(gs[1])

    # if no range is given, use the full range
    if xrange is None:
        xrange = (np.min(x_data), np.max(x_data))

    if yrange is None:
        yrange = (np.min(y_data), np.max(y_data))

    # first bin the data and plot it as a colormesh
    histo_2d, xedges, yedges = np.histogram2d(x_data, y_data, bins=[bins,bins],
                                               range=[xrange, yrange],
                                               normed=False)

    histo_2d = np.transpose(histo_2d)
    plt.pcolormesh(xedges, yedges, histo_2d, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])

    clevels = conf_level(histo_2d, conf_contours)
    #print clevels
    linestyles = []
    colors = []
    fmtdict = {}
    for (clevel, conf_label) in zip(clevels, conf_labels):
        linestyles.append('--')
        colors.append('green')
        fmtdict[clevel] = conf_label

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    histo_2d_smooth = ndimage.gaussian_filter(histo_2d, sigma=1., order=0)
    cs = plt.contour(histo_2d_smooth, extent=extent, levels=clevels,
                     linestyles=linestyles, colors=colors, linewidths=2)

    plt.clabel(cs, fmt=fmtdict, inline=True, fontsize=20)

    plt.xlim(xrange)
    plt.ylim(yrange)

    histo_x = np.histogram(x_data, bins=hbins, range=xrange, normed=True)[0]
    histo_y = np.histogram(y_data, bins=hbins, range=yrange, normed=True)[0]
    # Restore positions lost by binning.
    x_vec = xrange[0] + (xrange[1] - xrange[0]) * \
             np.array(range(0, len(histo_x)))/float(len(histo_x) - 1)

    y_vec = yrange[0] + (yrange[1] - yrange[0]) * \
             np.array(range(0, len(histo_y)))/float(len(histo_y) - 1)

    # plot the marginalized x data
    plt.subplot(gs[3])
    plt.plot(x_vec, histo_x, '-', lw=3, color='black', ls='steps')
    plt.ticklabel_format(style="sci", axis='x', scilimits=(1,2))
    plt.xticks(fontsize=16)
    plt.yticks([])
    plt.xlabel(r'$%s$' % xlabel, fontsize=24)
    #plt.ylabel(r'$\cal L$', fontsize=24)
    plt.xlim(xrange)
    plt.ylim(0.0, 1.1*np.max(histo_x))

    # plot the marginalized y data
    plt.subplot(gs[0])
    plt.plot(histo_y, y_vec, '-', lw=3, color='black', ls='steps')
    plt.ticklabel_format(style="sci", axis='y', scilimits=(1,2))
    plt.yticks(fontsize=16)
    plt.xticks([])
    #plt.xlabel(r'$\cal L$', fontsize=24)
    plt.ylabel(r'$%s$' % ylabel, fontsize=24)
    plt.xlim(0.0, 1.1*np.max(histo_y))
    plt.ylim(yrange)

    plt.savefig(filename, format="eps")
    #plt.show()


def plot_chains(filename, nsigma=3.):
    np.set_printoptions(threshold='nan')
    mcmc_data = h5py.File(filename, "r")
    chain_data = mcmc_data['chain']
    desc_data = mcmc_data['desc']

    print chain_data.keys(), "plot range (nsigma): ", nsigma
    #print chain_data['moment_1'].value 

    for pair in itertools.combinations(chain_data, 2):
        x_data = chain_data[pair[0]].value
        y_data = chain_data[pair[1]].value
        filename = "joint_dist_%s_with_%s.eps" % (pair[0], pair[1])
        print filename
        (x_mean, y_mean) = (np.mean(x_data), np.mean(y_data))
        (x_std, y_std) = (np.std(x_data), np.std(y_data))

        
        #print (x_mean, y_mean), (x_std, y_std)
        xrange = [x_mean - nsigma * x_std, x_mean + nsigma * x_std]
        yrange = [y_mean - nsigma * y_std, y_mean + nsigma * y_std]
        plot_2d_histo(x_data, y_data, xrange=xrange, yrange=yrange,
                      xlabel=desc_data[pair[0]].value,
                      ylabel=desc_data[pair[1]].value,
                      bins=50, filename=filename)

    mcmc_data.close()


def main():
    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")

    parser.add_option("-s", "--nsigma",
                      action="store",
                      dest="nsigma",
                      default=3.,
                      help="Range (in sigma) of axes")

    (options, args) = parser.parse_args()
    optdict = vars(options)

    if len(args) != 1:
        parser.error("wrong number of arguments")

    #print options, args

    plot_chains(args[0], nsigma=optdict['nsigma'])


if __name__ == '__main__':
    main()
