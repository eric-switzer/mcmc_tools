#!/usr/bin/env python
r"""Various tools to calculate the entropy either from covariance or samples"""
import numpy as np
import scipy.special
from scipy import spatial
import math
from optparse import OptionParser
import ast


def safe_det_dim(matrix_in):
    r"""Return the dimensions and det of a matrix
    'safe' because it works with scalars; can add more
    """
    as_array = np.atleast_2d(matrix_in)
    return (np.linalg.det(as_array), as_array.shape[0])


def analytic_entropy_normal(covmat):
    r"""analytically find the entropy of a normal distribution"""
    (detval, dim) = safe_det_dim(covmat)
    # factor is 2 pi e
    return 0.5 * np.log( 17.0794684453**float(dim) * detval )


def multivariate_normal_distribution(mean, covmat, samples):
    r"""Calculate the multivariate normal over a set of samples"""
    (detval, dim) = safe_det_dim(covmat)
    prefactor = 1. / (2. * math.pi)**(float(dim) / 2.) / math.sqrt(detval)

    dvec = samples - mean
    covinv = np.linalg.inv(covmat)
    return prefactor * np.exp(-0.5 * np.sum(dvec * np.dot(covinv, dvec.T).T,
                              axis=1))


def nearest_1dfast(samples, orig_sort=True, kind="quicksort"):
    r"""Calcluate the 1D nearest neighbors using sort

    orig_sort can be used to ensure the nearest neighbor distances match the
    indices of the inputs. If only the distribution of distances is needed,
    ignore this.
    """
    sort_ind = np.argsort(samples, kind=kind)
    sorted_samp = samples[sort_ind]

    deltas = np.abs(np.roll(sorted_samp, 1) - sorted_samp)
    rdeltas = np.roll(deltas, -1)

    outvec = np.array([min(pair) for pair in zip(deltas, rdeltas)])

    if orig_sort:
        unsort_ind = np.zeros_like(sort_ind)
        unsort_ind[sort_ind] = range(len(sort_ind))
        outvec = outvec[unsort_ind]

    return outvec


def nearest_vector_1d(samples):
    r"""find the 1D separation between nearest neighbors using brute force
    here `samples` is the vector of positions"""
    outvec = np.zeros_like(samples)
    for (sample, idx) in zip(samples, range(len(samples))):
        delta_arr = np.abs(samples - sample)
        outvec[idx] = np.min(delta_arr[delta_arr != 0])

    return outvec


def sample_entropy_nn(samples, multiplicity=None):
    r"""find the entropy of the sample using nearest neighbors
    Citation: Nearest Neighbor Estimates of Entropy
    Singh et al. Am. J. Mathmetical and Management Science

    Use kdtrees or sort to find the nearest neigh. dist.

    >>> dim = 2
    >>> nsamp = 50000
    >>> mean = np.zeros(dim)

    >>> covmat = np.array([[60., 5.4],[5.4, 10.7]])
    >>> samples = np.random.multivariate_normal(mean, covmat, nsamp)
    >>> result = sample_entropy_nn(samples)/analytic_entropy_normal(covmat)
    >>> np.testing.assert_almost_equal(result, 1., decimal=2)

    >>> variance = 11.
    >>> samples = np.random.normal(0, np.sqrt(variance), nsamp)
    >>> result = sample_entropy_nn(samples)/analytic_entropy_normal(variance)
    >>> np.testing.assert_almost_equal(result, 1., decimal=2)

    """
    try:
        ndim = float(samples.shape[1])
    except IndexError:
        ndim = 1.

    if ndim > 1:
        tree = spatial.KDTree(samples)
        # need to find nearest 2 because nearest is point itself
        nn_dist = tree.query(samples, k=2)[0][:, 1]
    else:
        nn_dist = nearest_1dfast(samples)

    # now pare out values at zero (duplicates in chain)
    whnzero = np.where(nn_dist > 0.)
    nn_dist = nn_dist[whnzero]

    ln_nn_dist = np.log(nn_dist)
    nsamp = float(len(nn_dist))

    if multiplicity is not None:
        multiplicity = multiplicity[whnzero]
        nsamp = float(np.sum(multiplicity))
        ln_nn_dist *= multiplicity

    factor = math.pi ** (ndim / 2.) / scipy.special.gamma(ndim / 2. + 1)
    result = ndim * np.sum(ln_nn_dist) / nsamp
    result += np.log(factor) + np.log(nsamp - 1.)
    result += 0.5772156649

    return result


def sample_entropy_file(filename, param_cols=None, mult_col=None):
    r"""Calculate the entropy of chains read from a file
    `param_cols` is the set of parameters column numbers to estimate over
    `mult_col` is an optional column giving chain multiplicies
    """
    in_file = open(filename, "r")
    all_data = np.genfromtxt(in_file, dtype="float")
    print "data dimensions: ", all_data.shape

    multiplicity = None
    if mult_col:
        multiplicity = all_data[:, mult_col].T
        print "multiplicity: ", multiplicity

    if param_cols:
        colarr = np.array(ast.literal_eval(param_cols))
        print "using parameter columns: ", colarr
        samples = all_data[:, colarr]
        if len(colarr) == 1:
            samples = samples[:, 0]
    else:
        samples = all_data

    print "using samples: ", samples
    print "entropy: ", sample_entropy_nn(samples, multiplicity=multiplicity)


def mcintegral_entropy(likelihood):
    r"""
    Given an appropriately-normalized likelihood, find the entropy

    >>> dim = 2
    >>> nsamp = 10000
    >>> mean = np.zeros(dim)
    >>> covmat = np.array([[60., 5.4],[5.4, 10.7]])

    >>> samples = np.random.multivariate_normal(mean, covmat, nsamp)
    >>> likelihood = multivariate_normal_distribution(mean, covmat, samples)
    >>> result = mcintegral_entropy(likelihood)
    >>> result /= analytic_entropy_normal(covmat)
    >>> np.testing.assert_almost_equal(result, 1., decimal=2)
    """
    norm = float(likelihood.shape[0])
    # MC integrand is -p ln p, so -p ln p / p
    return (1. / norm) * np.sum(-np.log(likelihood))


def main():
    r"""call sample_entropy_file on an output chain file"""
    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")

    parser.add_option("-c", "--cols",
                      action="store",
                      dest="param_cols",
                      default=None,
                      help="Column numbers for parameters")

    parser.add_option("-m", "--mult",
                      action="store",
                      dest="mult_col",
                      default=None,
                      help="Column number of multiplicity")

    (options, args) = parser.parse_args()
    optdict = vars(options)

    if len(args) != 1:
        parser.error("wrong number of arguments")

    sample_entropy_file(args[0], **optdict)


if __name__ == "__main__":
    #import doctest
    # run some tests
    #OPTIONFLAGS = (doctest.ELLIPSIS |
    #               doctest.NORMALIZE_WHITESPACE)
    #doctest.testmod(optionflags=OPTIONFLAGS)

    main()
