#!/usr/bin/env python
r"""Take mcmc chain summaries and overplots their best-fitting models and
a band indicating 1-sigma fluctuations on models across the chain"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import h5py
from optparse import OptionParser
import wrap_emcee


def plot_best_fit(hd5_file_list, plot_filename="best_fit.png",
                  obs_axis_name="bands", format="png",
                  xlabel="x", ylabel="y", nobands=False,
                  xmin=0.1, xmax=17., nx=300, nsample=10000, threads=20):
    r"""Draw `nsample` parameters from a chain and evaluate the model function
    there, find the mean and stdev of the model over this chain, plot.
    """
    obs_axis = 10.**np.linspace(np.log10(xmin), np.log10(xmax),
                               num=nx, endpoint=True)

    model_means = []
    model_stdevs = []
    meas_means = []
    meas_stdevs = []
    meas_axes = []
    colors = ["red", "green", "blue", "orange"]

    for filename in hd5_file_list:
        mcmc_data = h5py.File(filename, "r")
        mcmc_params = mcmc_data['params']
        mcmc_defaults = mcmc_data['defaults']
        eval_model = wrap_emcee.evaluate_model(mcmc_data, obs_axis,
                                               obs_axis_name=obs_axis_name,
                                               nsample=nsample,
                                               threads=threads)

        model_means.append(np.mean(eval_model, axis=1))
        model_stdevs.append(np.std(eval_model, axis=1))
        meas_means.append(mcmc_params['meas_means'].value)
        meas_stdevs.append(mcmc_params['meas_cov'].value)
        meas_axes.append(mcmc_defaults[obs_axis_name].value)
        mcmc_data.close()

    colors = colors[0: len(model_means)]

    fig = plt.figure(1, figsize=(7,7))
    fig.subplots_adjust(hspace=0.001, wspace=0.001, left=0.2, bottom=0.1,
                        top=0.975, right=0.98)

    plt.xlim([xmin, xmax])
    plt.xlabel(r"$%s$" % xlabel, fontsize=18)
    plt.ylabel(r"$%s$" % ylabel, fontsize=18)

    for (dmean, dstd, color) in zip(model_means, model_stdevs, colors):
        if not nobands:
            plt.fill_between(obs_axis, dmean - dstd,
                             dmean + dstd, facecolor=color,
                             interpolate=True, alpha=0.5, linewidth=0)

            plt.plot(obs_axis, dmean, linestyle="-", linewidth=1, color="black")
        else:
            plt.plot(obs_axis, dmean, linestyle="-", linewidth=2, color=color)

    # assume that all given chains are based on the same data
    print meas_means[0], np.diag(meas_stdevs[0])
    plt.errorbar(meas_axes[0], meas_means[0],
                 yerr=np.sqrt(np.diag(meas_stdevs[0])),
                 fmt="o", linewidth=2, color="black")

    plt.savefig(plot_filename, format=format)



def main():
    r"""parse arguments to plot_best_fit from the command line"""
    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")

    parser.add_option("-o", "--plotname",
                      action="store",
                      dest="plot_filename",
                      default="best_fit.png",
                      help="Output filename")

    parser.add_option("-f", "--format",
                      action="store",
                      dest="format",
                      default="png",
                      help="Output format")

    parser.add_option("-a", "--axis_name",
                      action="store",
                      dest="obs_axis_name",
                      default="bands",
                      help="parameter name specifying the axis of observations")

    parser.add_option("-x", "--xlabel",
                      action="store",
                      dest="xlabel",
                      default="x",
                      help="x label")

    parser.add_option("-y", "--ylabel",
                      action="store",
                      dest="ylabel",
                      default="y",
                      help="y label")

    parser.add_option("-l", "--xmin",
                      action="store",
                      dest="xmin",
                      type="float",
                      default=0.1,
                      help="Minimum x to plot")

    parser.add_option("-r", "--xmax",
                      action="store",
                      dest="xmax",
                      type="float",
                      default=17.,
                      help="Maximum x to plot")
    
    parser.add_option("-n", "--nx",
                      action="store",
                      dest="nx",
                      type="int",
                      default=300,
                      help="Number of x points")

    parser.add_option("-s", "--nsample",
                      action="store",
                      dest="nsample",
                      type="int",
                      default=10000,
                      help="Number of samples from chain to eval. model on")

    parser.add_option("-t", "--nthreads",
                      action="store",
                      dest="threads",
                      type="int",
                      default=20,
                      help="Number of threads to use in model eval.")

    parser.add_option("-b", "--nobands",
                      action="store_true",
                      dest="nobands",
                      default=False,
                      help="Do not show error bands")

    (options, args) = parser.parse_args()
    optdict = vars(options)

    if len(args) < 1:
        parser.error("wrong number of arguments")

    if len(args) > 4:
        parser.error("can not overlay more than 4 best fits")

    plot_best_fit(args, **optdict)


if __name__ == '__main__':
    main()
