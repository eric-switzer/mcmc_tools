r"""wrap_emcee: convenient bookkeeping for emcee

The emcee sampler tracks parameters to be estimated in the walker chains and
additional arguments to specify the likelihood. In many problems, these may
need to interchange fluidly to fix parameters, test sensitivity, etc. This
wrapper does the bookkeeping for chain vs. fixed variables and saves the output
chains by their variable names (with order preserving correlations in the
chain). ERS 22Aug2012.
"""
import numpy as np
import emcee
import h5py


def lnprob_generic(theta, params):
    r"""Log-likelihood, normal distribution; optional prior

    `theta` is the vector of parameters to vary
    params = { 'mu': mean values of the measurement
               'icov': C^-1 of the measurement
               'model_func': model function to call
               'theta_prior': optional prior on means of parameters
               'icov_prior': C^-1 of the optional prior
               'var_table': mapping from theta to model parameters
               'def_kwarg': default keyword args of the model
               'verbose': print diagnostic quantities
             }

    `var_table` is a dictionary that connects variables in theta to parameters
    in the function to fit. The parameters are stores in `def_kwarg`.
    The model is that the model function is called with **def_kwarg. Any of
    these parameters that are in var_table will be overwritten by the active
    values in the chain in theta.
    """
    def_kwarg = params['def_kwarg']
    var_table = params['var_table']

    # replace the default values with the chain variables
    for var_item in var_table:
        def_kwarg[var_item] = theta[var_table[var_item]]

    model_output = params['model_func'](**def_kwarg)

    diff = model_output - params['mu']
    chisq = -np.dot(diff, np.dot(params['icov'], diff)) / 2.
    if params['verbose']:
        print "likelihood: ", theta, model_output, params['mu'], diff, chisq

    if 'theta_prior' in params:
        pdiff = params['theta_prior'] - theta
        chisq -= np.dot(pdiff, np.dot(params['icov_prior'], pdiff)) / 2.
        if params['verbose']:
            print "prior: ", params['theta_prior'], theta, pdiff, chisq

    return chisq


def function_wrapper(funcname):
    r"""return a function based on its string name
    This is convenient for calling functions between modules.
    """

    funcattr = None
    funcsplit = funcname.split(".")

    # consider http://docs.python.org/dev/library/importlib.html
    if len(funcsplit) > 1:
        mod = __import__(".".join(funcsplit[0:-1]))
        for comp in funcsplit[1:-1]:
            mod = getattr(mod, comp)

        funcattr = getattr(mod, funcsplit[-1])
    else:
        funcattr = globals()[funcsplit[0]]

    print funcname, funcattr
    return funcattr


def call_mcmc(meas_means, meas_cov, default_params, fit_list, model_funcname,
             theta_prior=None, icov_prior=None,
             nwalkers=250, threads=1, verbose=False,
             outfile="save_chain.hd5"):
    r"""
    `meas_means` is the mean value of the measurement
    `meas_cov` is the covariance matrix of these measurements
    `default_params` is the dict of keyword arguments for the model function
    `fit_list` is the list of parameters in default params to let float:
    each entry of the list is a dict specifying the nature of the fit,
    fit_entry = {"kwarg_name": is the name in default params
                 "kwarg_desc": is a descriptive name of the variable
                 "walker_range": range over which to uniformly dist. walkers
                 "prior_mean": (optional) central value of a prior
                 "prior_std": (optional) std. dev. of the prior (ignoring cov)
                }
    The list is ordered in the same way as the variables are packed into the
    parameter variable `theta` of the chain. The prior on parameters should
    follow the same ordering scheme.
    `theta_prior` and `icov_prior` use this to specify a mean and covariance
    for a prior on parameters; this overwrites information from the fit_list,
    which does not support covariance in the prior, but is more convenient.
    """

    # find the list of variables to run the chain over
    # initialize the walkers; these just need to be informed guesses
    ndim = 0            # dimension of parameters in chain
    walker_start = {}   # the initial position of the walkers
    var_desc = {}       # description of the chain variables
    var_table = {}      # a table relating the array index to kwarg name
    prior_info = {}     # mean/std prior on some chain variables
    for fit_item in fit_list:
        kwarg_name = fit_item['kwarg_name']
        if kwarg_name in default_params:
            var_desc[kwarg_name] = fit_item['kwarg_desc']

            try:
                prior_info[kwarg_name] = [fit_item['prior_mean'],
                                          fit_item['prior_std']]
                print "%s using prior %s" % \
                                (kwarg_name, repr(prior_info[kwarg_name]))

            except KeyError:
                print "%s has no prior" % kwarg_name

            wleft = fit_item['walker_range'][0]
            wright = fit_item['walker_range'][1]

            walker_start[kwarg_name] = np.random.rand(nwalkers) * \
                                            (wright - wleft) + wleft

            var_table[kwarg_name] = ndim
            ndim += 1
        else:
            print "error: %s in chain not an argument of the model" % \
                  kwarg_name
            raise ValueError

    print var_table

    p0 = np.zeros((nwalkers, ndim))
    for var_item in var_table:
        p0[:, var_table[var_item]] = walker_start[var_item]

    meas_icov = np.linalg.inv(meas_cov)
    params = { 'mu': meas_means,
               'icov': meas_icov,
               'model_func': function_wrapper(model_funcname),
               'var_table': var_table,
               'def_kwarg': default_params,
               'verbose': verbose
             }

    # now determine the prior (if given)
    use_prior = False
    if (theta_prior is not None) and (icov_prior is not None):
        print "using given prior mean/cov; check indexing!"
        use_prior = True
    else:
        theta_prior = np.zeros(ndim)
        icov_prior = np.zeros((ndim, ndim))
        for kwarg_name in prior_info:
            use_prior = True
            ind = var_table[kwarg_name]
            theta_prior[ind] = prior_info[kwarg_name][0]
            icov_prior[ind, ind] = 1. / prior_info[kwarg_name][1]**2.

    if use_prior:
        print "using prior with mean = ", theta_prior
        print "prior inverse cov = ", icov_prior
        params['theta_prior'] = theta_prior
        params['icov_prior'] = icov_prior
    else:
        print "using no prior on paramters"

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_generic,
                                    args=[params], threads=threads)

    print "priming sampler"
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()

    print "using mean position: ", np.mean(pos, axis=0)

    print "starting run..."
    sampler.run_mcmc(pos, 1000)

    print "Acceptance fraction: %g" % np.mean(sampler.acceptance_fraction)

    summaryfile = h5py.File(outfile, "w")
    for kwarg_name in var_table:
        summaryfile[kwarg_name] = sampler.flatchain[:, var_table[kwarg_name]]
    summaryfile.close()

    # other output methods
    #pickle.dump(sampler, open('save.pkl', 'wb'))
    #pickle.dump(sampler.flatchain, open(outfile, 'wb'))

    print "chain mean: ", np.mean(sampler.flatchain, axis=0)
    print "chain std: ", np.std(sampler.flatchain, axis=0)
    print "chain cov: ", np.cov(np.transpose(sampler.flatchain))
    print "chain corrcoef: ", np.corrcoef(np.transpose(sampler.flatchain))

    return sampler.flatchain
