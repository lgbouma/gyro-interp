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
import warnings

from scipy.stats import norm, uniform

from gyroemp.models import slow_sequence_residual, slow_sequence
from gyroemp.age_scale import agedict

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
    Parallelize gyro_age_posterior calculation over a requested grid of
    ages.  The speedup from this function is not very good -- it gives
    only factor of 2x speedup for typical parameters.  If the task is to
    calculate gyro age posteriors for many stars, you will do better by
    parallelization over STARS, rather than AGES.  See e.g.,
    /drivers/run_prot_teff_grid.py.

    Args are as in gyro_age_posterior, but nworkers is either "max" or an
    integer number of cores to use.
    """

    # raise a traceback warning - this functoin should rarely be used.
    # TODO: remove it entirely?
    raise UserWarning(
        "The speedup from parallel_gyro_age_posterior is not very good."
    )
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
    Given a star's true age evaluate the gyro posterior over the given
    grids in Teff and y=Prot-μ.
    """

    (age, verbose, gaussian_teff, Prot, Prot_err, teff_grid,
     y_grid, bounds_error, n, reference_ages
    ) = task

    assert y_grid.ndim == 1

    if verbose:
        print(age)
        print(f"{datetime.now().isoformat()} begin 0")

    gaussian_Prots = []

    resid_obs_grid = Prot - slow_sequence(
        teff_grid, age, verbose=False, bounds_error=bounds_error, n=n,
        reference_ages=reference_ages
    )

    assert resid_obs_grid.ndim == 1

    gaussian_Prots = gaussian_pdf_broadcaster(
        y_grid, resid_obs_grid, Prot_err
    )

    if verbose:
        print(f"{datetime.now().isoformat()} end 0")
        print(f"{datetime.now().isoformat()} begin 1")

    # probability of a given residual (data-model) over dimensions of
    # y_grid X Teff_grid
    resid_y_Teff = slow_sequence_residual(
        age, y_grid=y_grid, teff_grid=teff_grid, verbose=False,
        bounds_error=bounds_error, n=n, reference_ages=reference_ages
    )

    if verbose:
        print(f"{datetime.now().isoformat()} end 1")
        print(f"{datetime.now().isoformat()} begin 2")

    integrand = resid_y_Teff * gaussian_Prots * gaussian_teff[None, :]

    p_age = np.trapz(np.trapz(integrand, teff_grid, axis=1), y_grid)

    if verbose:
        print(f"{datetime.now().isoformat()} end 2")

    return p_age


def gyro_age_posterior(
    Prot, Teff, Prot_err=None, Teff_err=None,
    age_grid=np.linspace(0, 2600, 500),
    verbose=False, bounds_error='limit', N_grid=256, n=0.5, age_scale='default'
    ):
    """
    Given a stellar rotation period and effective temperature, as well as their
    optional uncertainties (all ints or floats), calculate the probability of a
    given age assuming the models.slow_sequence_residual model holds.

    If Prot_err and Teff_err are not given, they are assumed to be 1% relative
    and 50 K, respectively.  These are best-case defaults.  The suggested
    effective temperature scale is implemented in gyrointerp.teff, in the
    given_dr2_BpmRp_AV_get_Teff_Curtis2020 function.  This assumes you have an
    accurate estimate for the reddening.  Spectroscopic temperatures are likely
    the next-best option, though a 2% systematic uncertainty floor is expected
    for them (see Tayar+2022, 2022ApJ...927...31T).

    Args:

        Prot, Teff, Prot_err, Teff_err: ints or floats, units of days and
        degrees Kelvin.  Must be positive.

        age_grid (np.ndarray): grid over which the age posterior is evaluated,
        units are fixed to be Myr.  A fine choice if bounds_error == 'limit' is
        np.linspace(0, 2600, 500).

        bounds_error: "limit" or "nan".  If "nan" ages below the minimum
        reference age return strings as described below.  If "limit", they
        return a prior-dominated number useable as an upper limit, based on the
        limiting rotation period at the closest cluster.  Default is "limit".

        N_grid (int): the dimension of the grid in effective
        temperature and reisdual-period over which the integration is
        performed to evaluate the posterior.  Default 256 to maximize
        performance.  Cutting down by factor of two leads to questionable
        convergence.

        n (int or float): assume Prot ~ t^{n} scaling

        age_scale (str): "default", "1sigmaolder", or "1sigmayounger".  Shifts
        the entire age scale appropriately, based on the user's beliefs about
        what ages of reference clusters are correct.  The scale is as described
        in the manuscript, and defined in /gyroemp/age_scale.py

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
    if Prot <= 0 or Teff <= 0:
        return np.nan*np.ones(len(age_grid))

    for param in [Prot_err, Teff_err]:
        if isinstance(param, (int, float)):
            if param <=0:
                return np.nan*np.ones(len(age_grid))

    if Prot_err is None:
        Prot_err = 0.01 * Prot

    if Teff_err is None:
        Teff_err = 50

    if N_grid < 256:
        print("WARNING! N_grid must be >=256 to get converged results.")

    assert isinstance(n, (int, float))

    assert age_scale in ["default", "1sigmaolder", "1sigmayounger"]

    ages = agedict[age_scale]
    reference_ages = agedict[age_scale]['reference_ages']

    #
    # special return cases
    #
    Prot_pleiades = slow_sequence(
        Teff, ages['Pleiades'], n=n, reference_ages=reference_ages
    )
    Prot_ngc6811 = slow_sequence(
        Teff, ages['NGC-6811'], n=n, reference_ages=reference_ages
    )
    Prot_rup147 = slow_sequence(
        Teff, ages['Rup-147'], n=n, reference_ages=reference_ages
    )

    if Prot < Prot_pleiades and Teff > 5000 and bounds_error == 'nan':
        return f"<{ages['Pleiades']} Myr"
    if Prot < Prot_pleiades and Teff <= 5000 and bounds_error == 'nan':
        return f"<{ages['NGC-6811']} Myr"
    if Prot > Prot_rup147 and bounds_error == 'nan':
        return f"<{ages['Rup-147']} Myr"

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
            y_grid, bounds_error, n, reference_ages
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
