"""
Main drivers:
    | ``gyro_age_posterior``
    | ``gyro_age_posterior_list``
    | ``gyro_age_posterior_mcmc``
"""
#############
## LOGGING ##
#############

import logging
from gyrointerp import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception

#############
## IMPORTS ##
#############
import os, pickle
from glob import glob
from gyrointerp.paths import DATADIR, CACHEDIR
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from numpy import array as nparr
from os.path import join
import warnings
from scipy.stats import norm

from gyrointerp.models import slow_sequence_residual, slow_sequence
from gyrointerp.age_scale import agedict
from gyrointerp.helpers import get_summary_statistics

from datetime import datetime
import multiprocessing as mp

def _agethreaded_gyro_age_posterior(
    Prot, Teff, Prot_err=None, Teff_err=None,
    age_grid=np.linspace(0, 3000, 500),
    verbose=True,
    bounds_error='4gyrlimit',
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
    LOGWARNING(
        "WRN! The speedup from _agethreaded_gyro_age_posterior is not good."
    )
    #
    # handle input parameters
    #
    if Prot_err is None:
        Prot_err = 0.01 * Prot

    if Teff_err is None:
        Teff_err = 100

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
        LOGINFO(age)
        LOGINFO(f"{datetime.now().isoformat()} begin 0")

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
        LOGINFO(f"{datetime.now().isoformat()} end 0")
        LOGINFO(f"{datetime.now().isoformat()} begin 1")

    # probability of a given residual (data-model) over dimensions of
    # y_grid X Teff_grid
    resid_y_Teff = slow_sequence_residual(
        age, y_grid=y_grid, teff_grid=teff_grid, verbose=False,
        bounds_error=bounds_error, interp_method=interp_method, n=n,
        reference_ages=reference_ages, popn_parameters=popn_parameters
    )

    if verbose:
        LOGINFO(f"{datetime.now().isoformat()} end 1")
        LOGINFO(f"{datetime.now().isoformat()} begin 2")

    integrand = resid_y_Teff * gaussian_Prots * gaussian_teff[None, :]

    p_age = np.trapz(np.trapz(integrand, teff_grid, axis=1), y_grid)

    if verbose:
        LOGINFO(f"{datetime.now().isoformat()} end 2")

    return p_age


def gyro_age_posterior(
    Prot, Teff, Prot_err=None, Teff_err=None,
    age_grid=np.linspace(0, 3000, 500),
    interp_method='pchip_m67', bounds_error='4gyrlimit', n=None,
    N_grid='default', age_scale='default', popn_parameters='default',
    verbose=False
    ):
    """
    Given a single star's rotation period and effective temperature, and
    (optionally) their uncertainties, what is the posterior probability
    distribution for its age?

    The answer returned by this code assumes that the
    ``gyrointerp.models.slow_sequence_residual`` model holds, which is the
    probability distribution described in BPH23 for the distribution of
    rotation periods at any given age and temperature.

    If Prot_err and Teff_err are not specified, they are assumed to be 1%
    relative and 100 K, respectively.  Spectroscopic temperature are
    acceptable.  The preferred photometric effective temperature
    scale is implemented in ``gyrointerp.teff``, in the
    ``given_dr2_BpmRp_AV_get_Teff_Curtis2020`` function.  This requires an
    accurate estimate for the reddening.  Whatever your effective
    temperature scale, it should ideally be compared against that in Appendix A
    of `Curtis+2020
    <https://ui.adsabs.harvard.edu/abs/2020ApJ...904..140C/abstract>`_.

    Args:

        Prot (int or float):
            Rotation period in units of days.

        Prot_err (int or float):
            Rotation period uncertainty in units of days.

        Teff: (int or float):
            Effective temperature in units of degrees Kelvin.  Must be between
            3800 and 6200 K.

        Teff_err (int or float):
            Effective temperature uncertainty in units of degrees Kelvin.

        age_grid (np.ndarray):
            Grid over which the age posterior is evaluated; units here and
            throughout are fixed to be megayears (10^6 years).  A fine choice
            is 500 points uniformly distributed between 0 and 3000 Myr:
            ``np.linspace(0, 3000, 500)``, assuming that the default choices
            of ``bounds_error == '4gyrlimit'`` and ``interp_method ==
            'pchip_m67'`` are being used.  If you wish to derive ages out to 4
            Gyr (the current upper boundary of plausibility, set by the age of
            M67), then set age_grid to ``np.linspace(0, 5000, 500)``, and set
            ``bounds_error == '4gyrextrap'``.

        interp_method (str):
            How will you interpolate between the polynomial fits to the
            reference open clusters? "pchip_m67" is the suggested default
            method, which uses Piecewise Cubic Hermite Interpolating
            Polynomials (PCHIP) to interpolate over not only 0.08-2.6 Gyr, but
            also sets the gradient in Prot vs Time in the 1-2.6 Gyr interval
            based on the observations of M67 from `Barnes+2016
            <https://ui.adsabs.harvard.edu/abs/2016ApJ...823...16B/abstract>`_
            and `Dungee+2022
            <https://ui.adsabs.harvard.edu/abs/2022ApJ...938..118D/abstract>`_.
            This yields an evolution of the rotation period envelope that is
            smooth and by design fits the cluster data from the age of
            alpha-Per through M67.  Other available interpolation methods
            include "skumanich_vary_n", "alt", "diff", "skumanich_fix_n_0.XX",
            "1d_linear", "1d_slinear", "1d_quadratic", and "1d_pchip", some of
            which are described in Appendix A of BPH23.   Unless you know what
            you are doing, "pchip_m67" is recommended.

        bounds_error (str):
            How will you extrapolate at the oldest and youngest clusters?  By
            default, this means at <0.08 Gyr and >4 Gyr.  Available options are
            "nan", "limit", "4gyrlimit", and "4gyrextrap".  Default
            "4gyrlimit" behavior at the old end is to not extrapolate at all, which
            means that this choice yields biased uncertainties at >~3.5 Gyr.
            If "4gyrextrap" is instead used, this will extrapolate by returning
            the rotation period linearly extrapolated from the Prot vs time
            slope at any temperature at 4 Gyr.  At the young end, the ansatz
            for both methods is that the period does not change.  In detail,
            this is physically wrong; the posterior in this regime is formally
            an age upper limit.  Finally, if "nan", ages above or below the
            minimum reference age return NaNs.

        n (None, int, or float):
            Power-law index analogous to the Skumanich braking index, but
            different in detail (read the source code to see how).  This is
            used only if ``interp_method == "alt"`` or ``interp_method ==
            "diff"``, neither of which is recommended for most users.  Default
            is None.

        N_grid (str or int):
            The number of grid points in effective temperature and
            residual-period over which the integration is performed to evaluate
            the posterior (Equation 1 of BPH23).  "default" sets it to ``N_grid
            = (Prot_grid_range)/Prot_err``, where "Prot_grid_range" is set to
            20 days, the range of the grid used in the integration.   This
            default ensures convergence, because numerical tests show
            convergence down to ~0.7x smaller grid sizes.  If an integer is
            passed, will use that instead.  For most users, "default" is best.

        age_scale (str):
            "default", "1sigmaolder", or "1sigmayounger".  Shifts the entire
            age scale appropriately, based on the user's beliefs about what
            ages of reference clusters are correct.  The scale is as described
            in the systematic uncertainty tests of BPH23, and defined in
            ``gyrointerp.age_scale``.

        popn_parameters (str):
            "default", or (dict) containing the population-level free
            parameters.  Keys of "a0", "a1", "k0", "k1", "y_g", "l_hidden", and
            "k_hidden" must all be specified.  Wrapped by
            ``gyro_age_posterior_mcmc``, for users who wish to do the
            population-level hyperparameter sampling described by BPH23.

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
        Teff_err = 100

    if Teff_err < 50:
        LOGWARNING(
            "WARNING: Teff uncertainties below 50 K are probably overly "
            "optimistic. Only do this if you have good reason to."
        )

    if Prot_err < 0.03:
        LOGWARNING(
            "WARNING: imposing period uncertainty floor of 0.03d "
            "The purpose of this floor is to prevent N_grid "
            "from growing too large.  This is justified "
            "because the gyro model has no information content "
            "on this scale."
        )
        Prot_err = 0.03

    if max(age_grid) > 4000 and bounds_error != '4gyrextrap':
        LOGWARNING(
            f"WARNING: Your age grid has a maximum of {max(age_grid)} "
            f"but you set bounds_error = {bounds_error}.  This can give "
            f"biased uncertainties at the old end.  You can fix this by "
            f"setting bounds_error to '4gyrextrap', which will give non-biased "
            f"uncertainties out to 4 Gyr.  Please do not try to use this code "
            f"to derive ages for stars older than 4 Gyr; it is not calibrated "
            f"in that age regime."
        )

    assert isinstance(N_grid, (int, str))

    Prot_grid_range = 20 # see y_grid below
    Teff_grid_range = 6200 - 3800

    if N_grid == "default":

        N_grid_0 = int(Prot_grid_range / Prot_err)
        N_grid_1 = int(Teff_grid_range / Teff_err)

        N_grid = max([N_grid_0, N_grid_1])

        if verbose:
            msg = (
                f"{datetime.now().isoformat()} "
                f"Prot {Prot}, Teff {Teff}, N_grid {N_grid}"
            )
            LOGINFO(msg)

    # Numerical tests for convergence (/tests/test_gyro_posterior_grid.py)
    # indicate we can go a little below the default grid resolution and still
    # do OK.  If they fail, raise an assertion error.
    N_grid_required_0 = 0.7*int(Prot_grid_range / Prot_err)
    N_grid_required_1 = 0.7*int(Teff_grid_range / Teff_err)
    if N_grid < N_grid_required_0:
        msg = (
            f"ERROR! N_grid must be >~{N_grid_required_0} to get "
            f"converged results.  Prot_err={Prot_err}"
        )
        raise AssertionError(msg)
    if N_grid < N_grid_required_1:
        msg = (
            f"ERROR! N_grid must be >~{N_grid_required_1} to get "
            f"converged results. Teff_err={Teff_err}"
        )
        raise AssertionError(msg)

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

    (Prot, Teff, Prot_err, Teff_err, star_id, age_grid, outdir, bounds_error,
     interp_method, n, age_scale, parameters, N_grid) = task

    Protstr = f"{float(Prot):.4f}"
    Teffstr = f"{float(Teff):.1f}"
    typestr = 'limitgrid'
    if parameters == 'default':
        paramstr = "_defaultparameters"
    else:
        paramstr = "_" + (
            repr(parameters).
            replace(' ',"_").replace("{","").replace("}","").
            replace("'","").replace(":","").replace(",","")
        )

    if star_id is None:
        sid = ''
    else:
        sid = f'{star_id}_'

    cachepath = os.path.join(
        outdir, f"{sid}Prot{Protstr}_Teff{Teffstr}_{typestr}{paramstr}.csv"
    )
    if not os.path.exists(cachepath):
        age_post = gyro_age_posterior(
            Prot, Teff, Prot_err=Prot_err, Teff_err=Teff_err,
            age_grid=age_grid, bounds_error=bounds_error,
            interp_method=interp_method, verbose=False, n=n,
            age_scale=age_scale, popn_parameters=parameters, N_grid=N_grid
        )
        df = pd.DataFrame({
            'age_grid': age_grid,
            'age_post': age_post
        })
        outpath = cachepath.replace(".csv", "_posterior.csv")
        df.to_csv(outpath, index=False)
        LOGINFO(f"Wrote {outpath}")

        d = get_summary_statistics(age_grid, age_post)
        d['Prot'] = Prot
        d['Teff'] = Teff
        df = pd.DataFrame(d, index=[0])
        df.to_csv(cachepath, index=False)
        LOGINFO(f"Wrote {cachepath}")

        return cachepath

    else:
        LOGINFO(f"Found {cachepath}")
        return cachepath


def _get_pop_samples(N_pop_samples):

    from gyrointerp.helpers import (
        get_population_hyperparameter_posterior_samples
    )
    flat_samples = get_population_hyperparameter_posterior_samples()

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
    Given the rotation period and effective temperature of a single star,
    sample over the population-level hyperparameters a1/a0, ln k0, ln k1, and
    y_g to determine the posterior probability distribution of the age.
    These are the dotted lines in the upper panel of Fig3 in BPH23.

    Parallelization is done over the hyperparameters.  However, the
    computational cost for a given star is about 1000x that of running the
    best-fit hyperparameters, as implemented in ``gyro_age_posterior``.  Use of
    this function is therefore generally not recommended, unless you have an
    understood need for doing things this way.

    Arguments are as in ``gyro_age_posterior``, but with four additions:

    Args:

        cache_id (str):
            The output posteriors will be cached to
            ``~/.gyrointerp_cache/{cache_id}`` (required).

        N_pop_samples (int):
            The number of draws from the posteriors for the aforementioned
            parameters to average over.

        N_post_samples (int):
            For each of the above draws, the number of draws from the resulting
            age posterior to cache before concatenating them all together.

        cachedir (str):
            Path to directory where individual posteriors will be cached (for
            every star, `N_post_samples` files will be written here!).  It is
            highly recommended that you specify this.

    Returns:

        np.ndarray : p_ages

            Numpy array containing posterior probabilities at each point of the
            age_grid.  Values are NaN if Teff outside of [3800, 6200] K.
    """

    assert isinstance(cachedir, str)
    if not os.path.exists(cachedir): os.mkdir(cachedir)

    #
    # calculate the individual posteriors for each set of population parameters
    #
    star_id = None
    popn_parameter_list = _get_pop_samples(N_pop_samples)

    tasks = [(Prot, Teff, Prot_err, Teff_err, star_id, age_grid, cachedir,
              bounds_error, interp_method, n, age_scale, paramdict, N_grid)
              for paramdict in popn_parameter_list]

    N_tasks = len(tasks)
    LOGINFO(f"Got N_tasks={N_tasks}...")
    LOGINFO(f"{datetime.now().isoformat()} begin")

    nworkers = mp.cpu_count()
    maxworkertasks= 1000

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)
    cachepaths = pool.map(_one_star_age_posterior_worker, tasks)
    pool.close()
    pool.join()

    LOGINFO(f"{datetime.now().isoformat()} end")

    #
    # draw the samples, and numerically construct the final pdf.
    #
    cachepaths = [c.replace(".csv", "_posterior.csv") for c in cachepaths]

    LOGINFO(42*'-')
    LOGINFO(f"Got {len(cachepaths)} cachepaths (from {len(tasks)})...")
    LOGINFO(42*'-')

    if len(cachepaths) < len(tasks):
        LOGINFO(42*'-')
        LOGINFO('WARNING! Got fewer outputs than expected.')
        LOGINFO(42*'-')

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
    cache_id, Prots, Teffs, Prot_errs=None, Teff_errs=None,
    star_ids=None, age_grid=np.linspace(0, 3000, 500), N_grid='default',
    bounds_error='4gyrlimit', interp_method='pchip_m67', nworkers=None
    ):
    """
    Given rotation periods and effective temperatures for many stars, run them
    in parallel.  This is a thin wrapper to ``gyro_age_posterior`` assuming
    default parameters.

    Args:

        cache_id (str):
            The output posteriors will be cached to
            ``~/.gyrointerp_cache/{cache_id}`` (required).

        Prots (np.ndarray):
            1-D array of rotation periods

        Teffs (np.ndarray):
            1-D array of temperatures, same length as Prots

        Prot_errs (np.ndarray):
            1-D array of rotation period uncertainties.  If None, assumes 1%
            relative uncertainties by default.

        Teff_errs (np.ndarray):
            1-D array of effective temperature uncertainties.  If None, assumes
            100K by default.

        star_ids (np.ndarray of strings):
            Arbitrary strings naming your stars; optional.  For example, if you
            give "TIC1234567", posteriors will be written to CSV files with a
            pattern matching
            ``TIC1234567_ProtXX.XXXX_TeffYYYY.Y_limitgrid_defaultparameters.csv``.
            If None, then the identifier is omitted.

        nworkers (int or None):
            Number of workers to thread over.  By default, will be taken to be
            all available CPU cores.

    Returns:

        List of paths to all available output posteriors at
        ``~/.gyrointerp_cache/{cache_id}``.  If you re-use your *cache_id*, this
        means you will get more than you asked for!
    """

    assert len(Prots) == len(Teffs)

    if Prot_errs is None:
        Prot_errs = 0.01*Prots

    if Teff_errs is None:
        Teff_errs = 100*np.ones(len(Prots))

    if star_ids is None:
        star_ids = [None]*len(Prots)
    else:
        assert len(star_ids) == len(Prots)

    outdir = CACHEDIR
    if not os.path.exists(outdir): os.mkdir(outdir)

    outdir = os.path.join(CACHEDIR, cache_id)
    if not os.path.exists(outdir): os.mkdir(outdir)

    n = None
    age_scale = "default"
    hyperparameters = "default"

    tasks = [(_prot, _teff, _proterr, _tefferr, _star_id, age_grid, outdir,
              bounds_error, interp_method, n, age_scale, hyperparameters,
              N_grid)
             for _prot, _teff, _proterr, _tefferr, _star_id
             in zip(Prots, Teffs, Prot_errs, Teff_errs, star_ids)]

    N_tasks = len(tasks)
    LOGINFO(f"Got N_tasks={N_tasks}...")
    LOGINFO(f"{datetime.now().isoformat()} beginning gyro_age_posterior_list")

    if nworkers is None:
        nworkers = mp.cpu_count()

    maxworkertasks= 1000

    os.environ['NUMEXPR_MAX_THREADS'] = str(nworkers)
    os.environ['NUMEXPR_NUM_THREADS'] = str(nworkers)

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    results = pool.map(_one_star_age_posterior_worker, tasks)

    pool.close()
    pool.join()

    LOGINFO(f"{datetime.now().isoformat()} end gyro_age_posterior_list")

    csvpaths = (
        glob(os.path.join(outdir, "*posterior.csv"))
    )

    LOGINFO(f"Returning N={len(csvpaths)} paths to posteriors.")

    return csvpaths
