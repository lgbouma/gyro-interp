"""
Run MCMC to get best-fit a1, y_g, logk0, logk1, logf, and their uncertainties.

Usage:
    Check modelid and grid matches what you want.  Then:
    $ python -u run_emcee_fit_gyro_model.py &> logs/logname.log &
"""

import emcee, corner
import multiprocessing as mp
import matplotlib as mpl
mpl.use('agg')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import pickle, os, corner
from collections import OrderedDict
from os.path import join
from scipy.optimize import minimize, Bounds

from gyrointerp.paths import RESULTSDIR, LOCALDIR
from gyrointerp.plotting import _get_model_histogram

def _get_data():

    datasets = OrderedDict()

    model_ids = ['120-Myr', '300-Myr', 'Praesepe']
    ages = [120, 300, 670]

    for age, model_id in zip(ages, model_ids):

        csvpath = os.path.join(RESULTSDIR, 'cdf_fast_slow_ratio',
                               f'{model_id}_cdf_fast_slow_ratio_data.csv')
        df = pd.read_csv(csvpath)
        data_midpoints = np.array(df.Teff_midpoints)
        data_ratio = np.array(df.ratio)
        f = 1/0.5323 # fudge factor to yield red-chi^2 near unity
        if age in [120, 300]:
            sigma = 0.1 * f**(-0.5) # uniform weighting across the 7 bins
        elif age == 670:
            sigma = 0.01 * f**(-0.5) # stricter requirement -- want it gonezo.
        else:
            raise NotImplementedError

        # x, y, y_err
        datasets[model_id] = [data_midpoints, data_ratio, sigma, age]

    return datasets


def log_probability(theta):

    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta)


def log_prior(theta):

    a1, y_g, logk0, logk1, logf = theta

    if (
        1 < a1 < 20 and
        0.1 < y_g < 1 and
        -10 < logk0 < 0 and
        -10 < logk1 < 0 and
        -3 < logf < 3
    ):
        return 0.0

    return -np.inf


def log_likelihood(theta):

    a1, y_g, logk0, logk1, logf = theta

    likelihoods = np.zeros(len(datasets))

    for ix, (name, (x, y, yerr, age)) in enumerate(datasets.items()):

        # get model, given age in Myr
        sigma_period = 0.51
        parameters = {
            'a0': 1,
            'a1': sample[0],
            'y_g': sample[1],
            'logk0': sample[2],
            'logk1': sample[3],
            'l_hidden': -2*sigma_period,
            'k_hidden': np.pi # a joke, but it works
        }

        h_vals_ss, h_vals_fs, teff_midway = _get_model_histogram(
            age, parameters=parameters
        )

        model_midpoints = teff_midway
        model_ratio = np.array( h_vals_fs / (h_vals_fs + h_vals_ss) )

        model = model_ratio*1.

        sigma2 = (np.exp(logf)*yerr)**2

        likelihoods[ix] = (
            -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        )

    return np.sum(likelihoods)


def main():

    modelid = "fitgyro_emcee_v02"
    OVERWRITE = 0 # whether to overwrite the MCMC samples for modelid
    n_steps = 32000 # number of MCMC steps.  10k is 100 minutes.  30k+2k burn.
    outdir = os.path.join(LOCALDIR, "gyrointerp", modelid)
    if not os.path.exists(outdir): os.mkdir(outdir)

    ############
    # get data #
    ############
    # make the data a global variable to speed up parallelization
    global datasets
    datasets = _get_data()

    ########################
    # get the MAP solution #
    ########################

    ndim = 5
    nwalkers = 32

    np.random.seed(42)

    CALC_MAP = False

    if CALC_MAP:
        nll = lambda *args: -log_likelihood(*args)

        eps = 1e-3
        #a1, y_g, logk0, logk1, logf = theta
        initial = np.array([8.7, 0.66, -5, -6.2, 1]) + eps*np.random.randn(ndim)

        bounds = bounds=(
            (1,20), (0.1,1), (-10, 5), (-10, 0), (-3, 3)
        )

        print('beginning minimization...')
        soln = minimize(nll, initial)

        # finds: array([ 8.25637368,  0.6727635 , -4.88458715, -6.23968725, -0.14829159])
        print(f"Got max-likelihood solution:")
        params = "a1, y_g, logk0, logk1, logf"
        print(params)
        print(soln.x)
        map_soln = soln.x

    else:
        map_soln = np.array(
            [8.25637486, 0.6727635, -4.8845869, -6.23968718, -0.14829162]
        )

    #######################################
    # fit the model and sample parameters #
    #######################################
    model_ids = ['120-Myr', '300-Myr', 'Praesepe']
    mstr = "_".join(model_ids)
    pklpath = os.path.join(outdir, f'fit_{mstr}.pkl')

    if OVERWRITE:
        if os.path.exists(pklpath): os.remove(pklpath)

    if not os.path.exists(pklpath):

        # Sample!
        pos = map_soln + 1e-4 * np.random.randn(nwalkers, ndim)
        nwalkers, ndim = pos.shape

        DO_PARALLEL = 1

        if DO_PARALLEL:
            nworkers = mp.cpu_count()
            with mp.Pool(nworkers) as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability,
                    pool=pool
                )
                sampler.run_mcmc(pos, n_steps, progress=True)

        if not DO_PARALLEL:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability
            )
            sampler.run_mcmc(pos, n_steps, progress=True)

        # autocorrelation time max is 91 steps for logk0. discard 20x that
        flat_samples = sampler.get_chain(discard=2000, flat=True)

        outdict = {
            'flat_samples': flat_samples
        }
        with open(pklpath, 'wb') as f:
            pickle.dump(outdict, f)
            print(f"Wrote {pklpath}")

        tau = sampler.get_autocorr_time()
        print(tau)

        outdir = os.path.join(RESULTSDIR, "emcee_fit_gyro_model")
        outtxt = os.path.join(outdir, "tau_estimate.txt")
        with open(outtxt, 'w') as f:
            f.writelines(repr(tau))
        print(f"Wrote {outtxt}")

    else:
        print(f"Found {pklpath}, loading.")
        d = pickle.load(open(pklpath, 'rb'))
        flat_samples = d['flat_samples']

    assert os.path.exists(pklpath)

    ##################
    # analyze output #
    ##################

    outdir = os.path.join(RESULTSDIR, "emcee_fit_gyro_model")
    if not os.path.exists(outdir): os.mkdir(outdir)

    # corner plot
    outpath = os.path.join(outdir, "corner.png")
    labels = ["a1", "y_g", "logk0", "logk1", "logf"]
    if not os.path.exists(outpath):
        fig = corner.corner(
            flat_samples, labels=labels
        )
        fig.savefig(outpath, bbox_inches="tight", dpi=300)
        print(f"Wrote {outpath}")

    # print best-fit parameters
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [15.9, 50, 84.1])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        print(txt)



if __name__ == "__main__":
    main()
