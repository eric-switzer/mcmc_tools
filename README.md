`mcmc_tools`
============

Several tools to help with MCMC outputs.
* `wrap_emcee`: bookkeeping for fixed vs. chain parameters in a model
* `plot_joint_dist.py`: tools to plot joint distributions
* `plot_model_bands.py`: evaluate the model over the chain; plot mean and band of fluctuations
* `relativistic_sz_fit.py`: example estimation problem
* `chain_entropy.py`: tools to measure the entropy of outputs

`emcee` and `h5py` need to be installed.

TODO:
* smoothed posterior dists (2d, 1d)
* log axes
* plotting lines at fixed parameter values

Used by (at least):
* `http://arxiv.org/abs/1211.3206` Sunyaev-Zeldovich signal processing and temperature-velocity moment method for individual clusters
* `http://arxiv.org/abs/1304.6121` Distinguishing different scenarios of early energy release with spectral distortions of the cosmic microwave background
