"""
Run MCMC to get best-fit C, C_y0, k0, k2, f, and their uncertainties.

Usage:
    Check modelid and grid matches what you want.  Then:
    $ python -u run_emcee_fit_gyro_model.py &> logs/logname.log &
"""

import emcee
import multiprocessing as mp
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import pickle, os, corner
from collections import OrderedDict
from os.path import join
from scipy.optimize import minimize, Bounds

from gyroemp.paths import RESULTSDIR, LOCALDIR

from gyroemp.plotting import _get_model_histogram

def log_probability(theta, datasets):

    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, datasets)


def log_prior(theta):

    C, C_y0, logk0, logk2, logf = theta

    if (
        1 < C < 20 and
        0.1 < C_y0 < 1 and
        -10 < logk0 < 5 and
        -10 < logk2 < 0 and
        -3 < logf < 3
    ):
        return 0.0

    return -np.inf


def log_likelihood(theta, datasets):

    C, C_y0, logk0, logk2, logf = theta

    likelihoods = np.zeros(len(datasets))

    for ix, (name, (x, y, yerr, age)) in enumerate(datasets.items()):

        # get model, given age in Myr
        sigma_period = 0.51
        parameters = {
            'A': 1,
            'B': 0,
            'C': C,
            'C_y0': C_y0,
            'logk0': logk0,
            'logk2': logk2,
            'l1': -2*sigma_period,
            'k1': np.pi # a joke, but it works
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

    modelid = "fitgyro_emcee_v00"
    outdir = os.path.join(LOCALDIR, "gyroemp", modelid)
    if not os.path.exists(outdir): os.mkdir(outdir)

    ############
    # get data #
    ############
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


    ########################
    # get the MAP solution #
    ########################

    ndim = 5
    nwalkers = 32
    n_steps = 5000

    np.random.seed(42)

    CALC_MAP = False

    if CALC_MAP:
        nll = lambda *args: -log_likelihood(*args)

        eps = 1e-3
        #C, C_y0, logk0, logk2, logf = theta
        initial = np.array([8.7, 0.66, -5, -6.2, 1]) + eps*np.random.randn(ndim)

        bounds = bounds=(
            (1,20), (0.1,1), (-10, 5), (-10, 0), (-3, 3)
        )

        print('beginning minimization...')
        soln = minimize(nll, initial, args=(datasets))

        C_ml, C_y0_ml, k0_ml, k2_ml, f_ml = soln.x

        # finds: array([ 8.25637368,  0.6727635 , -4.88458715, -6.23968725, -0.14829159])
        # or with bounds, array([ 8.68683268,  0.67138394, -4.96275124, -6.20941048, -0.16000444])
        print(f"Got max-likelihood solution:")
        params = "C, C_y0, logk0, logk2, logf"
        print(params)
        print(soln.x)
        map_soln = soln.x

    else:
        map_soln = np.array([ 8.68683268,  0.67138394, -4.96275124, -6.20941048, -0.16000444])

    #######################################
    # fit the model and sample parameters #
    #######################################
    mstr = "_".join(model_ids)
    pklpath = os.path.join(outdir, f'fit_{mstr}.pkl')

    if not os.path.exists(pklpath):

        # do everything!

        pos = map_soln + 1e-4 * np.random.randn(nwalkers, ndim)
        nwalkers, ndim = pos.shape

        #TODO TODO NEED TO AT LEAST MULTIHTREAD HERE!!!! OR OTHERWISE MAKE MUCH
        #FASTER
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(datasets,)
        )
        sampler.run_mcmc(pos, n_steps, progress=True)

        outdict = {
            'flat_samples': flat_samples
        }
        with open(pklpath, 'wb') as f:
            pickle.dump(outdict, f)
            print(f"Wrote {pklpath}")

        import IPython; IPython.embed()
        # TODO: check autocorrelation time...    maybe iteratively?
        tau = sampler.get_autocorr_time()
        print(tau)


    else:
        print(f"Found {pklpath}, loading.")
        d = pickle.load(open(pklpath, 'rb'))
        flat_samples = d['flat_samples']

    assert os.path.exists(pklpath)

    ##################
    # analyze output #
    ##################
    #FIXME


if __name__ == "__main__":
    main()
