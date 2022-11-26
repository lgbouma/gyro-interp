"""
Contents:
    gyro_age_posterior
    parallel_gyro_age_posterior

Under-the-hood:
    _gyro_age_posterior_worker
"""
import os, pickle
from gyroemp.paths import DATADIR
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from numpy import array as nparr
from os.path import join
from datetime import datetime

from scipy.stats import norm, uniform

from gyroemp.models import slow_sequence_residual, slow_sequence

import multiprocessing as mp

def parallel_gyro_age_posterior(
    Prot, Teff, Prot_err=None, Teff_err=None,
    age_grid=np.linspace(0, 2600, 500),
    verbose=True,
    bounds_error='limit',
    N_grid=256,
    nworkers='max',
):
    """
    Parallelize gyro_age_posterior calculation over the grid of requested ages.

    Note that the speedup from this function is not very good -- it gives only
    factor of 2x speedup for typical parameters.

    If the task is to calculate gyro age posteriors for many stars, you will do
    better by parallelization over STARS, rather than AGES.

    Args are as in gyro_age_posterior, but nworkers is either "max" or an
    integer number of cores to use.
    """
    #
    # handle input parameters
    #
    if Prot_err is None:
        Prot_err = 0.05

    if Teff_err is None:
        Teff_err = 50

    if N_grid < 256:
        print("WARNING! N_grid must be >=256 to get converged results.")

    #
    # special return cases
    #
    Prot_pleiades = slow_sequence(Teff, 120)
    Prot_ngc6811 = slow_sequence(Teff, 1000)
    Prot_rup147 = slow_sequence(Teff, 2600)

    if Prot < Prot_pleiades and Teff > 5000 and bounds_error == 'nan':
        return '<120 Myr'
    if Prot < Prot_pleiades and Teff <= 5000 and bounds_error == 'nan':
        return '<300 Myr'
    #if Prot > Prot_ngc6811 and bounds_error == 'nan':
    #    return '>1000 Myr'
    if Prot > Prot_rup147 and bounds_error == 'nan':
        return '>2600 Myr'

    #
    # calculate the posterior
    #

    # Define the gaussian probability density function in y (see
    # doc/20220919_width_of_slow_sequence.txt).  Have the grid
    # dimensions be 1 different from each other to ease debugging.
    teff_grid = np.linspace(3800, 6200, N_grid-1)
    y_grid = np.linspace(-14, 6, N_grid)

    gaussian_teff = norm.pdf(teff_grid, loc=Teff, scale=Teff_err)

    p_ages = []

    #
    # QUEUE different content from gyro_age_posterior...
    #

    if nworkers == 'max':
        nworkers = mp.cpu_count()

    maxworkertasks= 1000

    tasks = [
        (age, verbose, gaussian_teff, Prot, Prot_err, teff_grid, y_grid,
         bounds_error) for age in age_grid
    ]

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)
    results = pool.map(_gyro_age_posterior_worker, tasks)
    pool.close()
    pool.join()

    p_ages = np.vstack(results).flatten()

    return p_ages


def gaussian_pdf_broadcaster(x, mu, sigma):
    """
    Define a 2D gaussian pdf using array broadcasting when `x` and `mu` have
    different dimensions.
    """

    factor = 1 / (sigma*np.sqrt(2*np.pi))

    val = np.exp(
        -0.5 * ((x[:,None] - mu[None,:])/sigma)**2
    )

    return factor * val


def _gyro_age_posterior_worker(task):
    """
    Worker that, for a given true age (and given all the relevant grids)
    evaluates the gyro posterior.
    """

    (age, verbose, gaussian_teff, Prot, Prot_err, teff_grid,
     y_grid, bounds_error, n
    ) = task

    if verbose:
        print(age)

    if verbose:
        print(f"{datetime.now().isoformat()} begin 0")

    gaussian_Prots = []

    resid_obs_grid = Prot - slow_sequence(
        teff_grid, age, verbose=False, bounds_error=bounds_error, n=n
    )

    assert y_grid.ndim == 1
    assert resid_obs_grid.ndim == 1

    gaussian_Prots = gaussian_pdf_broadcaster(
        y_grid, resid_obs_grid, Prot_err
    )

    # # NOTE: comment below is the slower non-vectorized implementation of the
    # # gaussian_pdf_broadcaster call above.  In this mode "step 0" is the
    # # slowest of the three steps -- took 30msec on a typical call, vs 5 msec
    # # for step 1 and 1 msec for step 2.  In the vectorized
    # # gaussian_pdf_broadcaster implementation, step 0 is only 3 msec.

    # for teff, resid_obs in zip(teff_grid, resid_obs_grid):
    #     # residual over dimension of Teff_grid
    #     gaussian_Prot = norm.pdf(y_grid, loc=resid_obs, scale=Prot_err)
    #     gaussian_Prots.append(gaussian_Prot)
    # # y_grid x Teff_grid of gaussians, accounting for the observational
    # # uncertainty
    # gaussian_Prots = np.vstack(gaussian_Prots).T

    if verbose:
        print(f"{datetime.now().isoformat()} end 0")

    if verbose:
        print(f"{datetime.now().isoformat()} begin 1")
    # probability of a given residual (data-model) over dimensions of
    # y_grid X Teff_grid
    resid_y_Teff = slow_sequence_residual(
        age, y_grid=y_grid, teff_grid=teff_grid, verbose=False,
        bounds_error=bounds_error, n=n
    )
    if verbose:
        print(f"{datetime.now().isoformat()} end 1")

    if verbose:
        print(f"{datetime.now().isoformat()} begin 2")
    integrand = resid_y_Teff * gaussian_Prots * gaussian_teff[None, :]

    p_age = np.trapz(np.trapz(integrand, teff_grid, axis=1), y_grid)
    if verbose:
        print(f"{datetime.now().isoformat()} end 2")

    return p_age


def gyro_age_posterior(
    Prot, Teff, Prot_err=None, Teff_err=None,
    age_grid=np.linspace(0, 2600, 500),
    verbose=False,
    bounds_error='limit',
    N_grid=256,
    n=0.5
    ):
    """
    Given a stellar rotation period and effective temperature, as well as their
    optional uncertainties (all ints or floats), calculate the probability of a
    given age assuming the models.slow_sequence_residual model holds.

    If Prot_err and Teff_err are not given, they are assumed to be 0.05 days and
    50 K, respectively.

    Args:

        Prot, Teff, Prot_err, Teff_err: ints or floats, units of days and
        degrees Kelvin.

        age_grid: grid over which the age posterior is evaluated, units are
        fixed to be Myr.  A good choice if bounds_error == 'limit' is
        np.linspace(0, 2600, 500).

        bounds_error: "nan" or "limit".  If "nan" ages below the minimum
        reference age return strings as described below.  If "limit", they
        return a prior-dominated number useable as an upper limit, based on the
        limiting rotation period at the closest cluster.  Default is "limit".

        N_grid (int): the dimension of the grid in effective
        temperature and reisdual-period over which the integration is
        performed to evaluate the posterior.  Default 256 to maximize
        performance.  Cutting down by factor of two leads to questionable
        convergence.

    Returns:
        * NaN if Teff outside [3800, 6200] K
        * String of "<120 Myr" if Prot and Teff put the star below the Pleiades
          sequence, and the star is hotter than 5000 K.
        * String of "<300 Myr" if Prot and Teff put the star below the Pleiades
          sequence, and the star is cooler than 5000 K.
        * String of ">2600 Myr" if Prot and Teff put the star above the
          Ruprecht-147/NGC-6819 sequence.
        * Otherwise, returns np.ndarray containing posterior probabilities
          calculated over age_grid.
    """
    #
    # handle input parameters
    #
    if Prot_err is None:
        Prot_err = 0.05

    if Teff_err is None:
        Teff_err = 50

    if N_grid < 256:
        print("WARNING! N_grid must be >=256 to get converged results.")

    #
    # special return cases
    #
    Prot_pleiades = slow_sequence(Teff, 120, n=n)
    Prot_ngc6811 = slow_sequence(Teff, 1000, n=n)
    Prot_rup147 = slow_sequence(Teff, 2600, n=n)

    if Prot < Prot_pleiades and Teff > 5000 and bounds_error == 'nan':
        return '<120 Myr'
    if Prot < Prot_pleiades and Teff <= 5000 and bounds_error == 'nan':
        return '<300 Myr'
    #if Prot > Prot_ngc6811 and bounds_error == 'nan':
    #    return '>1000 Myr'
    if Prot > Prot_rup147 and bounds_error == 'nan':
        return '>2600 Myr'

    #
    # calculate the posterior
    #

    # Define the gaussian probability density function in y := Prot - μ.  Have
    # the grid dimensions be 1 different from each other to ease debugging.
    teff_grid = np.linspace(3800, 6200, N_grid-1)
    y_grid = np.linspace(-14, 6, N_grid)

    gaussian_teff = norm.pdf(teff_grid, loc=Teff, scale=Teff_err)

    p_ages = []
    for age in age_grid:

        task = (
            age, verbose, gaussian_teff, Prot, Prot_err, teff_grid,
            y_grid, bounds_error, n
        )
        p_age = _gyro_age_posterior_worker(task)

        p_ages.append(p_age)

    p_ages = np.vstack(p_ages).flatten()

    # return a normalized probability distribution.
    p_ages /= np.trapz(p_ages, age_grid)

    return p_ages


if __name__ == "__main__":
    # testing
    Prot = 6
    Teff = 5500
    age_grid = np.linspace(120, 1000, 50)
    age_post = gyro_age_posterior(Prot, Teff, age_grid=age_grid)
