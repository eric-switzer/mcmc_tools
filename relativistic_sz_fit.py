r"""Repeat the analysis of Prokhorov and Colafrancesco 2012 using an MCMC"""
import scipy.optimize as optimize
import numpy as np
import wrap_emcee
import h5py
# TODO: add nonlinear fitter an compare result to paper

def challinor_lasenby_sz(x_freq):
    r"""Return the g_0 (x), g_1 (x) and g_2 (x) functions from
    Challinor & Lasenby 1998, Eq. 28 and 33
    """
    # precalculate some common quantities
    exp_x = np.exp(x_freq)
    exp_xd2 = np.exp(x_freq / 2.)
    x_freq_sq = x_freq ** 2.

    c_x = x_freq * ((exp_x + 1.) / (exp_x - 1.))
    c_x2 = c_x ** 2.
    c_x3 = c_x ** 3.
    c_x4 = c_x ** 4.
    c_x5 = c_x ** 5.

    sinh_xd2_sq = ((exp_x - 1.) / (2. * exp_xd2))**2.

    prefactor = x_freq ** 4. * exp_x / (exp_x - 1.) ** 2.

    g0 = c_x - 4.
    g0 *= prefactor

    g1 = -10. + 47. / 2. * c_x - 42. / 5. * c_x2 + 7. / 10. * c_x3
    g1 += 7. / 5. * x_freq_sq / sinh_xd2_sq * (c_x - 3.)
    g1 *= prefactor

    g2 = -15. / 2. + 1023. / 8. * c_x - 868 / 5. * c_x2 + 329. / 5. * c_x3
    g2 += -44. / 5. * c_x4 + 11. / 30. * c_x5
    g2_paren = -2604. + 3948 * c_x - 1452 * c_x2 + 143. * c_x3
    g2 += x_freq_sq / 30. / sinh_xd2_sq * g2_paren
    g2 += x_freq ** 4. / 60. / sinh_xd2_sq ** 2 * (-528. + 187. * c_x)
    g2 *= prefactor

    return (g0, g1, g2)


def sz_model(bands=None, tau=None,
             moment_1=None, moment_2=None, moment_3=None):

    # if the result is non-physical, throw it out of the chain
    #print moment_1, moment_2, moment_3
    if (moment_1 < 0.) or \
       (moment_2 < 0.) or \
       (moment_3 < 0.) or \
       (tau < 0.):
        return -np.inf

    (g0, g1, g2) = challinor_lasenby_sz(bands)
    return tau * (moment_1 * g0 + moment_2 * g1 + moment_3 * g2)


def call_sz_fit(filename):
    # vector of central frequencies in x=h nu/k_B T_CMB
    meas_freq = np.array([150., 275., 600., 857.]) * 0.0176
    # measured flux and its errors in units of delta I (MJy/sr) / I_o
    meas_means = np.array([-0.325, 0.21, 0.268, 0.097]) / 269.914232952
    meas_err = np.array([0.015, 0.077, 0.031, 0.019]) / 269.914232952
    # fractional bandwidth of each flux measurement
    #meas_window = np.array([0.1, 0.1, 0.1, 0.1])
    meas_cov = np.diag(meas_err * meas_err)

    default_params = {}
    default_params['bands'] = meas_freq
    #defdefault_params['bandwidth'] = meas_window
    default_params['tau'] = 0.0138
    #default_params['tau'] = 1.
    default_params['moment_1'] = 0.
    default_params['moment_2'] = 0.
    default_params['moment_3'] = 0.


    tau_fit = {"kwarg_name": "tau",
               "kwarg_desc": "\\tau",
               "walker_range": [0, 0.014 * 2.],
               "prior_mean": 0.0138,
               "prior_std": 0.0016
              }

    tau_fit_noprior = {"kwarg_name": "tau",
                       "kwarg_desc": "\\tau",
                       "walker_range": [0, 0.014 * 2.],
                      }

    moment1_fit = {"kwarg_name": "moment_1",
                   "kwarg_desc": "\\langle T_e \\rangle/m_e c^2",
                   "walker_range": [0, 0.02 * 2.],
                  }

    moment2_fit = {"kwarg_name": "moment_2",
                   "kwarg_desc": "\\langle T_e^2 \\rangle/m_e^2 c^4",
                   "walker_range": [0, 7.e-4 * 2.],
                  }

    moment3_fit = {"kwarg_name": "moment_3",
                   "kwarg_desc": "\\langle T_e^3 \\rangle/m_e^3 c^6",
                   "walker_range": [0, 1.e-5 * 2.],
                  }

    #fit_list = [tau_fit, moment1_fit, moment2_fit, moment3_fit]
    fit_list = [moment1_fit, moment2_fit, moment3_fit]
    # TODO: why are those below ~2 times smaller?:
    #fit_list = [tau_fit, moment1_fit, moment2_fit]
    #fit_list = [moment1_fit, moment2_fit]

    chain_out = wrap_emcee.call_mcmc(meas_means, meas_cov, default_params,
                                    fit_list, "prokhorov_fit.sz_model",
                                    nwalkers=250, threads=15, verbose=False,
                                    outfile=filename)


def analyze_sz_fit(filename):
    mcmc_data = h5py.File(filename, "r")
    chain_data = mcmc_data['chain']

    sigma = 511. * np.sqrt( chain_data['moment_2'].value -
                            chain_data['moment_1'].value ** 2.)

    print np.mean(sigma[np.isfinite(sigma)]), np.std(sigma[np.isfinite(sigma)])


np.seterr(all='ignore')
if __name__ == '__main__':
    filename = "prokhorov.hd5"
    call_sz_fit(filename)
    analyze_sz_fit(filename)
