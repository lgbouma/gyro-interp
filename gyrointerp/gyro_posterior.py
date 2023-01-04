"""
Contents:
    | gyro_age_posterior
    | gyro_age_posterior_mcmc
    | gyro_age_posterior_list

Under-the-hood:
    | _gyro_age_posterior_worker
    | _one_star_age_posterior_worker
    | _get_pop_samples
    | _agethreaded_gyro_age_posterior
"""
import os, pickle
from glob import glob
from gyrointerp.paths import DATADIR, LOCALDIR
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from numpy import array as nparr
from os.path import join
import warnings

from scipy.stats import norm, uniform

from gyrointerp.models import slow_sequence_residual, slow_sequence
from gyrointerp.age_scale import agedict
from gyrointerp.helpers import given_grid_post_get_summary_statistics

from datetime import datetime
import multiprocessing as mp

def _agethreaded_gyro_age_posterior(
    Prot, Teff, Prot_err=None, Teff_err=None,
    age_grid=np.linspace(0, 3000, 500),
    verbose=True,
    bounds_error='limit',
    N_grid='default',
    nworkers='max',
):
    """
    Parallelize gyro_age_posterior calculation over a requested grid of
    ages.  The speedup from this function is not very good -- it gives
    only factor of 2x speedup for typical parameters.  If the task is to
    calculate gyro age posteriors for many stars, you will do better by
    parallelization over STARS, rather than AGES.  See e.g.,
    gyro_age_posterior_mcmc in this module, or /drivers/run_prot_teff_grid.py
    for examples.

    Args are as in gyro_age_posterior, but nworkers is either "max" or an
    integer number of cores to use.
    """

    # print a warning - this functoin should rarely be used.
    # TODO: remove it entirely?
    print(
        "WRN! The speedup from _agethreaded_gyro_age_posterior is not good."
    )
    #
    # handle input parameters
    #
    if Prot_err is None:
        Prot_err = 0.01 * Prot

    if Teff_err is None:
        Teff_err = 50

    if N_grid < 512:
        print("WARNING! N_grid must be >=512 to get converged results.")

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


def _gaussian_pdf_broadcaster(x, mu, sigma):
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

    (age, verbose, gaussian_teff, Prot, Prot_err, teff_grid, y_grid,
     bounds_error, interp_method, n, reference_ages, popn_parameters
    ) = task

    assert y_grid.ndim == 1

    if verbose:
        print(age)
        print(f"{datetime.now().isoformat()} begin 0")

    gaussian_Prots = []

    resid_obs_grid = Prot - slow_sequence(
        teff_grid, age, verbose=False, bounds_error=bounds_error,
        interp_method=interp_method, n=n, reference_ages=reference_ages
    )

    assert resid_obs_grid.ndim == 1

    gaussian_Prots = _gaussian_pdf_broadcaster(
        y_grid, resid_obs_grid, Prot_err
    )

    if verbose:
        print(f"{datetime.now().isoformat()} end 0")
        print(f"{datetime.now().isoformat()} begin 1")

    # probability of a given residual (data-model) over dimensions of
    # y_grid X Teff_grid
    resid_y_Teff = slow_sequence_residual(
        age, y_grid=y_grid, teff_grid=teff_grid, verbose=False,
        bounds_error=bounds_error, n=n, reference_ages=reference_ages,
        popn_parameters=popn_parameters
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
    age_grid=np.linspace(0, 3000, 500),
    verbose=False, bounds_error='4gyrlimit', interp_method='pchip_m67', n=None,
    N_grid='default', age_scale='default', popn_parameters='default'
    ):
    """
    Given a single star's rotation period and effective temperature, and the
    optional uncertainties (all ints or floats), calculate the probability of a
    given age assuming the ``gyrointerp.models.slow_sequence_residual`` model
    holds.

    If Prot_err and Teff_err are not given, they are assumed to be 1% relative
    and 50 K, respectively.  These are best-case defaults.  The suggested
    effective temperature scale is implemented in ``gyrointerp.teff``, in the
    ``given_dr2_BpmRp_AV_get_Teff_Curtis2020`` function.  This assumes you have
    an accurate estimate for the reddening.  Spectroscopic temperatures are
    the next-best option.

    Args:

        Prot, Prot_err : int or float.
            Units of days.  Must be positive.

        Teff, Teff_err : int or float.
            Units of degrees Kelvin.  Must be positive.

        age_grid : np.ndarray.
            Grid over which the age posterior is evaluated, units are fixed to
            be Myr (10^6 years).  A fine choice if ``bounds_error ==
            '4gyrlimit'`` and ``interp_method == 'pchip_m67'`` is
            np.linspace(0, 3000, 500).

        bounds_error : str
            "nan", "limit" or "4gyrlimit".  Extrapolation behaviors are as follows.
            If "limit", return the limiting rotation period at the closest
            cluster given in ``reference_model_ids``.  If "4gyrlimit",
            extrapolate out to 4 Gyr based on the ``reference_model_ids`` and
            the adopted interpolation method, regardless of where they
            truncate.  Past 4Gyr, take the same behavior as "limit".  If "nan",
            ages above or below the minimum reference age return nans

        interp_method : str
            Implemented interpolation methods include "skumanich_vary_n",
            "alt", "diff", "skumanich_fix_n_0.XX", "1d_linear", "1d_slinear",
            "1d_quadratic", "1d_pchip", and "pchip_m67".  The latter is the
            default method, because it yields evolution of rotation periods in
            time that are smooth and by design fit the cluster data from the
            age of alpha-Per through M67.  Unless you know what you are doing,
            "pchip_m67" is recommended".  "1d_linear", "1d_slinear", and
            "1d_quadratic" are as in ``scipy.interpolate.interp1d``.
            "1d_pchip" is ``scipy.interpolate.PchipInterpolator``.

        n : None, int, or float
            Power-law index analogous to the Skumanich braking index, but
            different in detail (see the implementation).  This is used only if
            ``interp_method == "alt"`` or ``interp_method == "diff"``, neither
            of which is recommended for most users.  Default is None.

        N_grid : str or int.
            The number of grid points in effective temperature and
            residual-period over which the integration is performed to evaluate
            the posterior.  "default" sets it to ``N_grid =
            (Prot_grid_range)/Prot_err``, where "Prot_grid_range" is set to 20
            days, the range of the grid used in the integration.   This default
            ensures convergence;  numerical tests show convergence down to
            ~0.7x smaller grid sizes.  If an integer is passed, will use that
            instead.

        age_scale : str.
            "default", "1sigmaolder", or "1sigmayounger".  Shifts the entire
            age scale appropriately, based on the user's beliefs about what
            ages of reference clusters are correct.  The scale is as described
            in the manuscript, and defined in /gyrointerp/age_scale.py

        popn_parameters : str.
            "default", or (dict) containing the population-level free
            parameters.  Keys of "a0", "a1", "k0", "k1", "y_g", "l_hidden", and
            "k_hidden" must all be specified.

    Returns:

        np.ndarray : p_ages

            Numpy array containing posterior probabilities at each point of the
            age_grid.  Values are NaN if Teff outside of [3800, 6200] K.
    """
    #
    # handle input parameters
    #
    if Prot <= 0 or Teff <= 0:
        return np.nan*np.ones(len(age_grid))

    if Teff < 3800 or Teff > 6200:
        return np.nan*np.ones(len(age_grid))

    for param in [Prot_err, Teff_err]:
        if isinstance(param, (int, float)):
            if param <=0:
                return np.nan*np.ones(len(age_grid))

    if Prot_err is None:
        Prot_err = 0.01 * Prot

    if Teff_err is None:
        Teff_err = 50

    if Teff_err < 50:
        print(
            "WARNING: Teff uncertainties below 50 K are probably overly "
            "optimistic. Only do this if you have good reason to."
        )

    if Prot_err < 0.03:
        print(
            "WARNING: imposing period uncertainty floor of 0.03d "
            "The purpose of this floor is to prevent N_grid "
            "from growing too large.  This is justified "
            "because the gyro model has no information content "
            "on this scale."
        )
        Prot_err = 0.03

    assert isinstance(N_grid, (int, str))

    if N_grid == "default":
        Prot_grid_range = 20 # see y_grid below
        N_grid = int(Prot_grid_range / Prot_err)
        if verbose:
            msg = (
                f"{datetime.now().isoformat()} "
                f"Prot {Prot}, Teff {Teff}, N_grid {N_grid}"
            )
            print(msg)

    # numerical tests for convergence (/tests/test_gyro_posterior_grid.py)
    # indicate we can go a little below the default grid resolution and still
    # do OK.
    N_grid_required = 0.7*int(Prot_grid_range / Prot_err)
    if N_grid < N_grid_required:
        print("WARNING! N_grid must be >~{N_grid_required} to get "
              "converged results.")

    Teff_grid_range = 6200 - 3800
    errmsg = "Default N_grid is too small because"
    assert Teff_grid_range / N_grid < Teff_err, errmsg

    assert isinstance(n, (int, float, type(None)))

    assert age_scale in ["default", "1sigmaolder", "1sigmayounger"]

    ages = agedict[age_scale]
    reference_ages = agedict[age_scale]['reference_ages']

    assert isinstance(popn_parameters, (str, dict))
    if isinstance(popn_parameters, str):
        assert popn_parameters == 'default'

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
            y_grid, bounds_error, interp_method, n, reference_ages,
            popn_parameters
        )
        p_age = _gyro_age_posterior_worker(task)

        p_ages.append(p_age)

    p_ages = np.vstack(p_ages).flatten()

    # return a normalized probability distribution.
    p_ages /= np.trapz(p_ages, age_grid)

    return p_ages


def _one_star_age_posterior_worker(task):

    (Prot, Teff, age_grid, outdir, bounds_error, interp_method, n, age_scale,
     parameters, N_grid) = task

    Protstr = f"{float(Prot):.4f}"
    Teffstr = f"{float(Teff):.1f}"
    typestr = 'limitgrid'
    bounds_error = 'limit'
    if parameters == 'default':
        paramstr = "_defaultparameters"
    else:
        paramstr = "_" + (
            repr(parameters).
            replace(' ',"_").replace("{","").replace("}","").
            replace("'","").replace(":","").replace(",","")
        )

    cachepath = os.path.join(
        outdir, f"Prot{Protstr}_Teff{Teffstr}_{typestr}{paramstr}.csv"
    )
    if not os.path.exists(cachepath):
        age_post = gyro_age_posterior(
            Prot, Teff, age_grid=age_grid, bounds_error=bounds_error,
            interp_method=interp_method,
            verbose=False, n=n, age_scale=age_scale,
            popn_parameters=parameters, N_grid=N_grid
        )
        df = pd.DataFrame({
            'age_grid': age_grid,
            'age_post': age_post
        })
        outpath = cachepath.replace(".csv", "_posterior.csv")
        df.to_csv(outpath, index=False)
        print(f"Wrote {outpath}")

        d = given_grid_post_get_summary_statistics(age_grid, age_post)
        d['Prot'] = Prot
        d['Teff'] = Teff
        df = pd.DataFrame(d, index=[0])
        df.to_csv(cachepath, index=False)
        print(f"Wrote {cachepath}")

        return cachepath

    else:
        print(f"Found {cachepath}")
        return cachepath


def _get_pop_samples(N_pop_samples):

    # TODO: cache this in an accessible way so that 
    # TODO: all users can use gyro_age_posterior_mcmc
    pklpath = os.path.join(LOCALDIR, "gyrointerp", "fitgyro_emcee_v02",
                           "fit_120-Myr_300-Myr_Praesepe.pkl")
    with open(pklpath, 'rb') as f:
        d = pickle.load(f)
        flat_samples = d['flat_samples']

    np.random.seed(42)
    sel_samples = flat_samples[
        np.random.choice(flat_samples.shape[0], N_pop_samples, replace=False)
    ]
    sigma_period = 0.51
    popn_parameter_list = []
    for ix in range(N_pop_samples):
        sample = sel_samples[ix, :]
        #a1, y_g, logk0, logk1, logf = theta
        parameters = {
            'a0': 1,
            'a1': sample[0],
            'y_g': sample[1],
            'logk0': sample[2],
            'logk1': sample[3],
            'l_hidden': -2*sigma_period,
            'k_hidden': np.pi # a joke, but it works
        }
        popn_parameter_list.append(parameters)

    return popn_parameter_list


def gyro_age_posterior_mcmc(
    Prot, Teff, Prot_err=None, Teff_err=None,
    age_grid=np.linspace(0, 3000, 500),
    verbose=False, bounds_error='4gyrlimit', interp_method='pchip_m67',
    N_grid='default', n=None, age_scale='default', N_pop_samples=512,
    N_post_samples=10000, cachedir=None
    ):
    """
    Same as gyro_age_posterior, but sampling over the population-level
    parameters a1/a0, ln k0, ln k1, and y_g.

    Arguments are as in gyro_age_posterior, but with three additions:

        N_pop_samples (int): the number of draws from the posteriors for the
        aforementioned parameters to average over.

        N_post_samples (int): for each of the above draws, the number of draws
        from the resulting age posterior to cache before concatenating them all
        together.

        cachedir (str): path to directory where individual posteriors will be
        cached (for every star, `N_post_samples` files will be written here!)

    Parallelizes over the population-parameters samples.  Takes 3 minutes to
    run `N_post_samples=512` on a 64 core machine.
    """

    assert isinstance(cachedir, str)
    if not os.path.exists(cachedir): os.mkdir(cachedir)

    #
    # calculate the individual posteriors for each set of population parameters
    #
    popn_parameter_list = _get_pop_samples(N_pop_samples)

    tasks = [(Prot, Teff, age_grid, cachedir, bounds_error, interp_method, n,
              age_scale, paramdict, N_grid) for paramdict in
              popn_parameter_list]

    N_tasks = len(tasks)
    print(f"Got N_tasks={N_tasks}...")
    print(f"{datetime.now().isoformat()} begin")

    nworkers = mp.cpu_count()
    maxworkertasks= 1000

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)
    cachepaths = pool.map(_one_star_age_posterior_worker, tasks)
    pool.close()
    pool.join()

    print(f"{datetime.now().isoformat()} end")

    #
    # draw the samples, and numerically construct the final pdf.
    #
    cachepaths = [c.replace(".csv", "_posterior.csv") for c in cachepaths]

    print(42*'-')
    print(f"Got {len(cachepaths)} cachepaths (from {len(tasks)})...")
    print(42*'-')

    if len(cachepaths) < len(tasks):
        print(42*'-')
        print('WARNING! Got fewer outputs than expected.')
        print(42*'-')

    age_samples = []
    for cachepath in cachepaths:
        df = pd.read_csv(cachepath)
        age_samples.append(
            df.sample(
                n=N_post_samples, replace=True, weights=df.age_post
            ).age_grid
        )
    age_samples = np.hstack(age_samples)

    denominator = len(age_samples)

    p_ages = []
    for age in age_grid:
        numerator = np.sum(np.isclose(age_samples, age))
        p_ages.append(numerator/denominator)

    return p_ages / np.trapz(p_ages, age_grid)


def gyro_age_posterior_list(
    cache_id, Prots, Teffs, age_grid=np.linspace(0, 3000, 500),
    N_grid='default', bounds_error='4gyrlimit', interp_method='pchip_m67'
    ):
    """
    Given lists of rotation periods and effective temperatures, run them in
    parallel.  This functions as a thin wrapper to gyro_age_posterior assuming
    default parameters.  The output posteriors will be cached to
    `$LOCALDIR/gyrointerp/{cache_id}`.

    Args:

    cache_id (str):
        string identifying the grid for cacheing

    Prots (np.ndarray):
        1-D array of rotation periods

    Teffs (np.ndarray):
        1-D array of temperatures, same length as Prots

    age_grid: as in gyro_age_posterior
    """

    assert len(Prots) == len(Teffs)

    outdir = os.path.join(LOCALDIR, "gyrointerp")
    if not os.path.exists(outdir): os.mkdir(outdir)

    outdir = os.path.join(LOCALDIR, "gyrointerp", cache_id)
    if not os.path.exists(outdir): os.mkdir(outdir)

    n = None
    age_scale = "default"
    hyperparameters = "default"

    tasks = [(_prot, _teff, age_grid, outdir, bounds_error, interp_method, n,
              age_scale, hyperparameters, N_grid)
             for _prot, _teff in zip(Prots, Teffs)]

    N_tasks = len(tasks)
    print(f"Got N_tasks={N_tasks}...")
    print(f"{datetime.now().isoformat()} begin")

    nworkers = mp.cpu_count()

    maxworkertasks= 1000

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    results = pool.map(_one_star_age_posterior_worker, tasks)

    pool.close()
    pool.join()

    print(f"{datetime.now().isoformat()} end")


if __name__ == "__main__":
    # testing
    Prot = 6
    Teff = 5500
    age_grid = np.linspace(0, 3000, 50)
    age_post = gyro_age_posterior(Prot, Teff, age_grid=age_grid)
