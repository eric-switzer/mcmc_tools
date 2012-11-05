#!/usr/bin/env python
r"""plot 2D histograms along combinations of variables
ERS Nov 2012
"""
import os
import sys
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


def find_range(data_in, nsigma=3., gap=0.3):
    r"""helper to find ranges of plots
    # TODO: make this more robust to outliers
    """
    delta = np.std(data_in)
    (dmin, dmax) = (np.min(data_in), np.max(data_in))

    if nsigma > 0.:
        center = np.mean(data_in)
        left = center - nsigma * delta
        right = center + nsigma * delta
    else:
        (left, right) = (dmin, dmax)

    gap = gap * delta
    left = max([left, dmin - gap])
    right = min([right, dmax + gap])

    return np.array([left, right])


def rescale_log(data_range, min_thresh=-2., max_thresh=3.):
    r"""rescale data for scientific notation"""
    largest = np.max(np.log10(np.abs(np.array(data_range / 10.))))

    if largest > min_thresh and largest < max_thresh:
        return None
    else:
        return int(largest)


def print_dict(data_dict, depth=""):
    r"""Print a tree represented as a dictionary"""
    for data_key in data_dict:
        current_depth = depth + " " * 4
        data_here = data_dict[data_key]
        if isinstance(data_here, dict):
            print "%s-%s:" % (current_depth, data_key)
            print_dict(data_here, current_depth)
        else:
            if isinstance(data_here, np.ndarray):
                if data_here.size > 10:
                    print "%s-> %s: shape %s" % (current_depth, data_key,
                                           data_here.shape)
                else:
                    print "%s-> %s: %s" % (current_depth, data_key,
                                           data_here)
            else:
                print "%s-> %s: %s" % (current_depth, data_key, data_here)


def pairname(pair):
    r"""give pairs of variables a uniform name convention"""
    return "%s_x_%s" % (pair[0], pair[1])


class PlotChain:
    """
    Class to handle plotting of joint distribution from MCMC chain outputs
    This is driven by the command line plot_chain below, but can also by called
    as a free-standing object.
    1. register the files, 2. produce the histograms, 3. get requested plots
    # TODO: extend varlist_and to varlist_or, plots including all variables
    # TODO: marginal axis off-by-one?
    # TODO: write code to summarize a chain output; percentiles; correlations
    """
    def __init__(self):
        # ranges and descriptions for variables in the chain
        self.file_info = {}
        self.file_list = []  # keep order
        # histogram information for each file
        self.histo_info = {}
        # plot range and description for variables in common
        self.plot_info = {}
        # variable list that all input files have in common
        self.varlist_and = None
        # base output plot filename
        # (default is to stack root names of each registered file)
        self.basename = None
        self.use_colormesh = False
        self.max_norm = False

    def process_chain_data(self, xbins=50, ybins=50, bins=100,
                           max_norm=True,
                           conf_contours=None,
                           conf_labels=None):
        r"""perform processing steps before making plots
        """
        # find the variables in common among several chains
        self.find_variables()
        self.find_histograms(xbins=xbins, ybins=ybins, bins=bins,
                             max_norm=max_norm,
                             conf_contours=conf_contours,
                             conf_labels=conf_labels)
        self.print_params()

    def register_file(self, filename, nsigma=3., rangegap=0.3, color="green"):
        r"""Read and hd5 chain output file and populate a table with its
        range and description information
        plot options:
            nsigma: if <0, use full range
            rangegap: leave a buffer of 1sigma*rangegap on either side
        """
        mcmc_data = h5py.File(filename, "r")
        chain_data = mcmc_data['chain']
        desc_data = mcmc_data['desc']

        plot_details = {}
        for varname in chain_data:
            var_info = {}
            var_chain = chain_data[varname].value
            var_info['shape'] = var_chain.shape
            var_info['range'] = find_range(var_chain, nsigma=nsigma,
                                           gap=rangegap)

            var_info['desc'] = desc_data[varname].value
            var_info['color'] = color
            varname_ascii = varname.encode('ascii', 'ignore')
            plot_details[varname_ascii] = var_info

        self.file_list.append(filename)
        self.file_info[filename] = plot_details
        self.histo_info[filename] = {}  # this gets filled in later
        mcmc_data.close()

        # make this the basename
        if self.basename is None:
            self.basename = os.path.splitext(filename)[0]
        else:
            self.basename += "_x_" + os.path.splitext(filename)[0]

    def print_params(self):
        r"""Print summaries of the various dictionaries"""

        print "Information from incoming files:"
        print "-" * 80
        print_dict(self.file_info)
        print "Final plot ranges and descriptions:"
        print "-" * 80
        print_dict(self.plot_info)
        print "Histogram data:"
        print "-" * 80
        print_dict(self.histo_info)

    def find_variables(self):
        r"""Generate the variables used in the plots
        1. find the variables in common in the input chains
        2. find the range that encloses all the common variables
        """
        # find the number of files and whether to use colormesh in plots
        num_files = len(self.file_list)
        if num_files == 0:
            print "no files have been loaded"
            sys.exit()

        if num_files > 1:
            print "multiple chain; using contours only"
            self.use_colormesh = False
        else:
            print "single chain; using contours and colormesh"
            self.use_colormesh = True

        for filename in self.file_list:
            newvars = self.file_info[filename].keys()
            if self.varlist_and is None:
                self.varlist_and = newvars
            else:
                self.varlist_and = [val for val in self.varlist_and \
                                    if val in newvars]

            self.varlist_and.sort()

        for variable_name in self.varlist_and:
            var_info = {}
            var_info['range'] = None
            var_info['desc'] = None
            for filename in self.file_list:
                range_here = self.file_info[filename][variable_name]["range"]
                desc_here = self.file_info[filename][variable_name]["desc"]
                if var_info['range'] is None:
                    var_info['range'] = range_here
                    var_info['desc'] = desc_here
                else:
                    newrange = np.zeros((2))
                    newrange[0] = min([var_info['range'][0], range_here[0]])
                    newrange[1] = max([var_info['range'][1], range_here[1]])
                    var_info['range'] = newrange

                    if desc_here != var_info['desc']:
                        print "descriptions mismatch, most recent: ", desc_here

                    var_info['desc'] = desc_here

            # now find the axis labels and rescaled axes (sci-notation)
            var_label = copy.deepcopy(var_info['desc'])
            var_info['multiplier'] = 1.
            var_info['plot_range'] = copy.deepcopy(var_info['range'])

            rescale_val = rescale_log(var_info['range'])
            if rescale_val:
                print "rescaling %s by %d" % (var_label, rescale_val)
                var_info['multiplier'] = 1. / 10. ** rescale_val
                var_info['plot_range'] *= var_info['multiplier']
                var_label += "\\cdot 10^{%d}" % -rescale_val

            # consider adding $\cal L$
            var_info["label"] = "$%s$" % "$ $".join(var_label.split("~"))

            self.plot_info[variable_name] = var_info

    def find_histograms(self, xbins=50, ybins=50, bins=100,
                        max_norm=True,
                        conf_contours=None, conf_labels=None):
        r"""Read the files and make histograms over the variable pairs"""
        if conf_contours == None:
            print "defaulting to 0.5, 0.95 enclosed"
            conf_contours = [0.5, 0.95]

        if conf_labels == None:
            conf_labels = ["$%d$" % int(conf * 100.) for conf in conf_contours]

        self.max_norm = max_norm

        for filename in self.file_list:
            mcmc_data = h5py.File(filename, "r")
            chain_data = mcmc_data['chain']
            self.histo_info[filename]["histo_pairs"] = {}
            self.histo_info[filename]["histo_vars"] = {}

            # find the 1D histograms
            for variable in self.varlist_and:
                var_data = copy.deepcopy(chain_data[variable].value)
                var_data *= self.plot_info[variable]['multiplier']
                varrange = self.plot_info[variable]['plot_range']
                varcolor = self.file_info[filename][variable]['color']

                print variable, np.mean(var_data), varrange
                (histo, edges) = np.histogram(var_data, bins=bins,
                                                 range=varrange,
                                                 normed=True)

                if max_norm:
                    histo /= np.max(histo)

                middle_vec = 0.5 * (edges[1:] + edges[:-1])
                histo_info = {"histo": histo,
                              "edges": edges,
                              "middle": middle_vec,
                              "color": varcolor}

                self.histo_info[filename]["histo_vars"][variable] = histo_info

            # find the 2D histograms over pairs
            for pair in itertools.combinations(self.varlist_and, 2):
                x_data = copy.deepcopy(chain_data[pair[0]].value)
                x_data *= self.plot_info[pair[0]]['multiplier']
                x_range = self.plot_info[pair[0]]['plot_range']
                x_color = self.file_info[filename][pair[0]]['color']

                y_data = copy.deepcopy(chain_data[pair[1]].value)
                y_data *= self.plot_info[pair[1]]['multiplier']
                y_range = self.plot_info[pair[1]]['plot_range']
                y_color = self.file_info[filename][pair[1]]['color']

                assert x_color == y_color, "colors for joint plot do not agree"

                histo_2d, xedges, yedges = np.histogram2d(x_data, y_data,
                                              bins=[xbins, ybins],
                                              range=[x_range, y_range],
                                              normed=False)

                # find the confidence regions
                # move this calculation up to the histogram code
                clevels = conf_level(histo_2d, conf_contours)
                linestyles = []
                colors = []
                fmtdict = {}
                for (clevel, conf_label) in zip(clevels, conf_labels):
                    if self.use_colormesh:
                        linestyles.append('--')
                    else:
                        linestyles.append('-')
                    colors.append(x_color)
                    fmtdict[clevel] = conf_label

                histo_info = {"histo": histo_2d,
                              "xedges": xedges,
                              "yedges": yedges,
                              "linestyles": linestyles,
                              "colors": colors,
                              "clevels": clevels,
                              "fmtdict": fmtdict}

                pair_str = pairname(pair)
                self.histo_info[filename]["histo_pairs"][pair_str] = \
                                                         histo_info

            mcmc_data.close()

    def plot_all_2d_histo(self, output_root=None,
                          color_scheme="binary",
                          file_format="eps"):
        r"""write each joint distribution to its own plot file"""
        if output_root is None:
            root = "./"
        else:
            root = output_root

        for pair in itertools.combinations(self.varlist_and, 2):
            pair_str = pairname(pair)
            plot_filename = "%s/%s-%s.%s" % \
                            (root, self.basename,
                             pair_str, file_format)

            self.plot_2d_histo(pair[0], pair[1],
                          plot_filename=plot_filename,
                          color_scheme=color_scheme,
                          file_format=file_format)

    def retrieve_histogram(self, x_var, y_var, fname):
        r"""retrieve a 2D histogram for the given variables/file
        This is mainly useful for when the transpose exists; e.g.
        given y_var, x_var but x_var, y_var is already calculated.
        TODO: simplify
        """
        pair_str = pairname((x_var, y_var))
        if pair_str in self.histo_info[fname]["histo_pairs"]:
            print "%s %s -> %s %s" % (x_var, y_var, x_var, y_var)
            histo_data = \
                copy.deepcopy(self.histo_info[fname]["histo_pairs"][pair_str])

        pair_str = pairname((y_var, x_var))
        if pair_str in self.histo_info[fname]["histo_pairs"]:
            print "%s %s -> %s %s" % (x_var, y_var, y_var, x_var)
            histo_data = \
                copy.deepcopy(self.histo_info[fname]["histo_pairs"][pair_str])
            plot_data = self.histo_info[fname]["histo_pairs"][pair_str]
            histo_data['histo'] = \
                        np.transpose(copy.deepcopy(plot_data["histo"]))
            histo_data['xedges'] = copy.deepcopy(plot_data["yedges"])
            histo_data['yedges'] = copy.deepcopy(plot_data["xedges"])

        return histo_data

    def plot_2d_histo(self, x_var, y_var,
                      plot_filename="plot_2d_histo.eps",
                      color_scheme="binary",
                      file_format="eps"):
        r"""Plot a joint distribution given x and y variable names
        TODO:? plt.ticklabel_format(style="sci", axis='x', scilimits=(1, 3))
        """
        pair_str = pairname((x_var, y_var))
        print "starting plot for vars %s: %s" % (pair_str, plot_filename)

        x_range = self.plot_info[x_var]['plot_range']
        y_range = self.plot_info[y_var]['plot_range']
        print "x_range %s, y_range %s" % (repr(x_range), repr(y_range))

        fig = plt.figure(1, figsize=(7, 7))

        fig.subplots_adjust(hspace=0.001, wspace=0.001,
                            left=0.15, bottom=0.12,
                            top=0.975, right=0.98)

        subplot_grid = gridspec.GridSpec(2, 2, width_ratios=[1, 4],
                                         height_ratios=[4, 1])

        plt.subplot(subplot_grid[1])
        plt.xticks([])
        plt.yticks([])
        plt.xlim(x_range)
        plt.ylim(y_range)
        first_run = True
        for filename in self.file_list:
            print "adding data from ", filename
            histo_data = self.retrieve_histogram(x_var, y_var, filename)
            histo_2d = np.transpose(histo_data["histo"])
            xedges = histo_data['xedges']
            yedges = histo_data['yedges']

            if self.use_colormesh:
                plt.pcolormesh(xedges, yedges, histo_2d,
                               cmap=plt.cm.get_cmap(color_scheme))

            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            histo_2d_smooth = ndimage.gaussian_filter(histo_2d,
                                                      sigma=1.5,
                                                      order=0)

            contours = plt.contour(histo_2d_smooth, extent=extent,
                                   levels=histo_data['clevels'],
                                   linestyles=histo_data['linestyles'],
                                   colors=histo_data['colors'],
                                   linewidths=2)

            # only label contours on one plot
            if first_run:
                plt.clabel(contours, fmt=histo_data['fmtdict'],
                           inline=True, fontsize=20)
                first_run = False

        # make the 1D marginalized x plots
        ax_x = plt.subplot(subplot_grid[3])
        plt.xticks(fontsize=16)
        if self.max_norm:
            plt.yticks([0., 0.25, 0.5, 0.75, 1.])
            ax_x.set_yticklabels([])
        else:
            plt.yticks([])

        plt.xlabel(self.plot_info[x_var]["label"], fontsize=24)
        plt.xlim(x_range)

        pdf_height = []
        for filename in self.file_list:
            plot_data = self.histo_info[filename]["histo_vars"][x_var]
            (x_vec, histo_x) = (plot_data['middle'], plot_data['histo'])

            plt.plot(x_vec, histo_x, '-', lw=3,
                     color=plot_data['color'])

            pdf_height.append(1.1 * np.max(histo_x))

        plt.ylim(0.0, max(pdf_height))

        # make the 1D marginalized y plots
        ax_y = plt.subplot(subplot_grid[0])
        plt.yticks(fontsize=16)
        if self.max_norm:
            plt.xticks([0., 0.25, 0.5, 0.75, 1.])
            ax_y.set_xticklabels([])
        else:
            plt.xticks([])

        plt.ylabel(self.plot_info[y_var]["label"], fontsize=24)
        plt.ylim(y_range)

        pdf_height = []
        for filename in self.file_list:
            plot_data = self.histo_info[filename]["histo_vars"][y_var]
            (y_vec, histo_y) = (plot_data['middle'], plot_data['histo'])

            plt.plot(histo_y, y_vec, '-', lw=3,
                     color=plot_data['color'])

            pdf_height.append(1.1 * np.max(histo_y))

        plt.xlim(0.0, max(pdf_height))

        plt.savefig(plot_filename, format=file_format)
        #plt.show()

    def plot_triangle(self, plot_filename="plot_triangle.eps",
                      varlist=None, size_multiplier=3.,
                      color_scheme="binary",
                      file_format="eps"):
        r"""Plot the triangle of joint parameter distribution"""
        print "making a joint param triangle plot: ", plot_filename
        if varlist is None:
            varlist = self.varlist_and

        nvars = len(varlist)
        print "number of vars = ", nvars

        plot_size = nvars * size_multiplier
        fig = plt.figure(1, figsize=(plot_size, plot_size))
        #fig.subplots_adjust(hspace=0.02, wspace=0.02, left=0.02, bottom=0.05,
        #                top=0.98, right=0.98)
        fig.subplots_adjust(hspace=0.02, wspace=0.02, left=0.02, bottom=0.08,
                            top=0.98, right=0.98)

        for x_ind in range(0, nvars):
            for y_ind in range(x_ind, nvars):
                print "plotting section: ", varlist[y_ind], varlist[x_ind]
                ax = plt.subplot2grid((nvars, nvars), (x_ind, y_ind))

                # if the plot is along the diagonal, plot marginalized
                if x_ind == y_ind:
                    m_var = varlist[x_ind]
                    if self.max_norm:
                        plt.yticks([0., 0.25, 0.5, 0.75, 1.])
                        ax.set_yticklabels([])
                    else:
                        plt.yticks([])

                    plt.xticks(fontsize=10)
                    plt.xlabel(self.plot_info[m_var]["label"], fontsize=20)
                    plt.xlim(self.plot_info[m_var]["plot_range"])
                    pdf_height = []
                    for filename in self.file_list:
                        plot_data = \
                            self.histo_info[filename]["histo_vars"][m_var]
                        m_vec = plot_data['middle']
                        m_histo = plot_data['histo']

                        # can also add ls="steps"
                        plt.plot(m_vec, m_histo, '-', lw=3,
                                 color=plot_data['color'])

                        pdf_height.append(1.1 * np.max(m_histo))

                    plt.ylim(0.0, max(pdf_height))

                else:  # plot the joint distribution
                    plt.xticks([])
                    plt.yticks([])
                    x_range = self.plot_info[varlist[x_ind]]['plot_range']
                    y_range = self.plot_info[varlist[y_ind]]['plot_range']
                    plt.xlim(x_range)
                    plt.ylim(y_range)
                    first_run = True
                    for filename in self.file_list:
                        histo_data = self.retrieve_histogram(varlist[x_ind],
                                                             varlist[y_ind],
                                                             filename)

                        histo_2d = histo_data["histo"]
                        xedges = histo_data['xedges']
                        yedges = histo_data['yedges']

                        if self.use_colormesh:
                            plt.pcolormesh(xedges, yedges, histo_2d,
                                           cmap=plt.cm.get_cmap(color_scheme))

                        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                        histo_2d_smooth = ndimage.gaussian_filter(histo_2d,
                                                                  sigma=1.5,
                                                                  order=0)

                        contours = plt.contour(histo_2d_smooth, extent=extent,
                                        levels=histo_data['clevels'],
                                        linestyles=histo_data['linestyles'],
                                        colors=histo_data['colors'],
                                        linewidths=2)

                        # only label contours on one plot
                        if first_run:
                            plt.clabel(contours, fmt=histo_data['fmtdict'],
                                       inline=True, fontsize=14)

                            first_run = False

        plt.savefig(plot_filename, format=file_format)


def plot_chains(filelist, plot_options):
    r"""Given an hd5 file written by wrap_emcee, plot the joint distribution
    of all the parameters in the output chains"""
    #   -list of hd5 files; if only one: black marginals, green contours
    # nsigma, filecolor, meshcolor, output, file_format, separate
    nfiles = len(filelist)
    if plot_options["filecolor"] is None:
        if nfiles == 1:
            plot_options["filecolor"] = ['black']
        else:
            plot_options["filecolor"] = ['green', 'blue', \
                                         'red', 'purple', 'black']

    plot_options["filecolor"] = plot_options["filecolor"][0: nfiles]

    chain_plot = PlotChain()
    print "plot range (nsigma): ", plot_options['nsigma']
    for (filename, color) in zip(filelist, plot_options["filecolor"]):
        chain_plot.register_file(filename, color=color,
                                 nsigma=plot_options['nsigma'])

    conf_contours = [float(confreg) for confreg in \
                     plot_options['confreg'].split(',')]

    max_norm = not plot_options['areanorm']

    print "using confidence regions: ", conf_contours
    chain_plot.process_chain_data(xbins=plot_options['nbins2d'],
                                  ybins=plot_options['nbins2d'],
                                  bins=plot_options['nbins1d'],
                                  max_norm=max_norm,
                                  conf_contours=conf_contours,
                                  conf_labels=None)

    if plot_options['separate']:
        chain_plot.plot_all_2d_histo(output_root=plot_options['output'],
                                     color_scheme=plot_options['meshcolor'],
                                     file_format=plot_options['file_format'])
    else:
        chain_plot.plot_triangle(plot_filename=plot_options['output'],
                                 color_scheme=plot_options['meshcolor'],
                                 file_format=plot_options['file_format'])


def main():
    r"""parse arguments to plot_chains from cmd line
    Consider adding:
        -histeps instead of lines for 1D distributions
        -B/W friendly traces
        -turn off confidence contour labels
    """
    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")

    parser.add_option("-n", "--nsigma",
                      action="store",
                      dest="nsigma",
                      default=3.,
                      help="Range (in sigma) of axes")

    parser.add_option("--nbins1d",
                      action="store",
                      dest="nbins1d",
                      default=100,
                      help="Number of bins in 1D histogram")

    parser.add_option("--nbins2d",
                      action="store",
                      dest="nbins2d",
                      default=50,
                      help="Number of bins in 2D histogram")

    parser.add_option("--confreg",
                      action="store",
                      dest="confreg",
                      default="0.5, 0.95",
                      help="Confidence regions (list of percentiles)")

    parser.add_option("--filecolor",
                      action="store",
                      dest="filecolor",
                      default=None,
                      help="List of colors for each file")

    parser.add_option("--meshcolor",
                      action="store",
                      dest="meshcolor",
                      default="binary",
                      help="Color scheme for distributions \
                            (gray/binary/jet are good options)")

    parser.add_option("-o", "--output",
                      action="store",
                      dest="output",
                      default=None,
                      help="Filename or output root (for -s)")

    parser.add_option("-f", "--format",
                      action="store",
                      dest="file_format",
                      default="eps",
                      help="File format (eps/png)")

    parser.add_option("-s", "--separate",
                      action="store_true",
                      dest="separate",
                      default=False,
                      help="Make separate plots for each joint var")

    parser.add_option("--areanorm",
                      action="store_true",
                      dest="areanorm",
                      default=False,
                      help="1D histograms normalized to 1 at maximum")

    (options, args) = parser.parse_args()
    optdict = vars(options)

    if len(args) < 1:
        parser.error("too few hd5 chain output files")

    #print options, args
    plot_chains(args, optdict)


if __name__ == '__main__':
    main()
