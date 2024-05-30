"""
Functions to fit rotation versus effective temperature sequences, or to quickly
return the results of those fits (including their interpolations!)

Contents:
    | ``reference_cluster_slow_sequence``
    | ``slow_sequence``
    | ``slow_sequence_residual``

Helper functions:
    | ``teff_zams``
    | ``g_lineardecay``
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
from gyrointerp.paths import DATADIR
import pandas as pd, numpy as np
from numpy import array as nparr
from os.path import join
from copy import deepcopy

from scipy.interpolate import interp1d, PchipInterpolator
from scipy.stats import norm, uniform

###########
# helpers #
###########
def teff_zams(age, bounds_error='limit'):
    """
    Physics-informed MIST effective temperature for the effective temperature a
    star has when it arrives on the ZAMS.  Changes over 80 to 1000 Myr (the
    floor beyond 1000 Myr is well below the 3800 K cool limit of BPH23).  For a
    test, see ``/tests/plot_teff_cuts.py``.
    """
    if isinstance(age, (float,int)):
        age = np.array([age])

    csvpath = os.path.join(
        DATADIR, "literature",
        "Choi_2016_MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_basic_arrival_times.csv"
    )
    df = pd.read_csv(csvpath)

    from scipy.interpolate import make_interp_spline

    # linear interpolation
    spl = make_interp_spline(df['age'], np.log10(df['Teff']), k=1)
    with np.errstate(divide='ignore'):
        teff0 = 10**spl(np.log10(age*1e6))

    max_teff = 10**spl(np.log10(80*1e6))
    min_teff = 10**spl(np.log10(1e9))

    bad = (age < 80) | (age > 1000)
    if bounds_error == 'nan':
        teff0[bad] = np.nan
    elif bounds_error == 'limit' or bounds_error in ['4gyrlimit', '4gyrextrap']:
        teff0[age < 80] = max_teff
        teff0[age > 1000] = min_teff

    return teff0


def _logistic(x, x0, L=1, k=0.1):
    """
    Logistic function; larger k makes the cutoff sharper
    """
    num = L
    denom = 1 + np.exp(-k * (x-x0))
    return num / denom


def _teff_0(age, bounds_error='4gyrlimit'):
    """
    Naive, by-eye midpoint for how the slow sequence taper moves with age.
    Defined for age from 120 to 1000 Myr.  If `bounds_error=='limit'`, then set
    as whatever the lowest and highest values are.

    Tested at /tests/plot_teff_cuts.py
    """

    if isinstance(age, (float,int)):
        age = np.array([age])

    # units: kelvin per Myr
    slope = (4500 - 4000) / (300 - 120)
    c = 4500

    teff0 = -(age-120) * slope + c

    bad = (age < 120) | (age > 1000)
    if bounds_error == 'nan':
        teff0[bad] = np.nan
    elif bounds_error == 'limit' or bounds_error in ['4gyrlimit', '4gyrextrap']:
        teff0[age < 120] = c
        teff0[age > 1000] = -(1000-120) * slope + c

    return teff0


def g_lineardecay(age, bounds_error='4gyrlimit', y_g=1/2):
    """
    Function *g(t)* from BPH23 defining the linear rate at which the uniform
    component amplitude from ``models.slow_sequence_residual`` decreases with
    age.  Unity at <=120 Myr, decreasing linearly to  `y_g` by 300 Myr (eg.
    1/2, 1/4, 1/6).  Decreases linearly thereafter, and once it reaches zero,
    it stays at zero.
    """

    if isinstance(age, (float,int)):
        age = np.array([age])

    # units: 1/Myr
    y1 = 1
    slope = (y1 - y_g) / (300 - 120)
    c = y1*1.

    c_uniform = -(age-120) * slope + c

    bad = (age < 120) | (age > 1000)

    if bounds_error == 'nan':
        c_uniform[bad] = np.nan
    elif bounds_error == 'limit' or bounds_error in ['4gyrlimit', '4gyrextrap']:
        c_uniform[age < 120] = c
        c_uniform[age > 1000] = -(1000-120) * slope + c

    negative = c_uniform < 0
    c_uniform[negative] = 0

    return c_uniform

########
# core #
########
def slow_sequence_residual(
    age,
    y_grid=np.linspace(-14, 6, 1000),
    teff_grid = np.linspace(3800, 6200, 1001),
    poly_order=7, n=None,
    reference_model_ids=[
        'α Per', '120-Myr', '300-Myr', 'Praesepe', 'NGC-6811', '2.6-Gyr', 'M67'
    ],
    reference_ages=[80, 120, 300, 670, 1000, 2600, 4000],
    popn_parameters='default',
    verbose=True,
    bounds_error='4gyrlimit', interp_method='pchip_m67'):
    """
    Given an effective temperature and an age, return the 2-D distribution of
    residuals around and underneath the slow sequence, sampled onto grids of
    *y_grid* X *teff_grid*, where *y_grid* is the residual of (rotation period
    data - mean gyrochronal model).  This is Equation 7 of BPH23.

    The two components of the residual are:

        * a gaussian in *y_grid* , with an age-varying cutoff in Teff
          (``teff_zams``), imposed as a logistic taper.

        * an age-varying and Teff-varying uniform distribution, multiplied by
          the inverse of the gaussian's taper (but with independent scale
          length), and then truncated to ensure that stars rotate faster than
          zero days, and to ensure that we model only the fast sequence.  This
          uniform distribution is also tapered by a logistic function at the
          "slow" end to yield a smoother transition to the gaussian.

    Args:
        teff_grid (np.ndarray):
            As in ``models.slow_sequence``.

        y_grid (np.ndarray):
            A grid over the residual of (rotation period - mean gyrochronal
            model).

        poly_order (int):
            As in ``models.slow_sequence``.

        reference_model_ids (list):
            As in ``models.slow_sequence``.

        reference_ages (list):
            As in ``models.slow_sequence``.

        interp_method (str):
            As in ``models.slow_sequence``.

        bounds_error (str):
            As in ``models.slow_sequence``.

        popn_parameters (str or dict):
            "default", or a dict containing the population-level free
            parameters.  Keys of "a0", "a1", "k0", "k1", "y_g", "l_hidden", and
            "k_hidden" must all be specified.

    Returns:

        np.ndarray : resid_y_Teff

            2d array with dimension (N_y_grid x N_teff_grid), containing the
            probability distribution of the residual over the dimensions of *y*
            and *Teff*.
    """

    assert len(reference_ages) == len(reference_model_ids)

    # The intrinsic width (RMS) of the slow sequence, in units of days.
    sigma_period = 0.51

    if popn_parameters == "default":
        # from run_emcee_fit_gyro_model; MAP-values
        # [8.25637486, 0.6727635, -4.8845869, -6.23968718, -0.14829162]
        a0 = 1
        a1 = 8.256
        y_g = 0.673
        k0 = np.e**-4.885
        k1 = np.e**-6.240
        l_hidden = -2*sigma_period
        k_hidden = np.pi # a joke, but it works

    elif isinstance(popn_parameters, dict):

        a0 = popn_parameters["a0"]
        a1 = popn_parameters["a1"]
        y_g = popn_parameters["y_g"]

        if "k0" in popn_parameters:
            k0 = popn_parameters["k0"]
        elif "logk0" in popn_parameters:
            k0 = np.exp(popn_parameters["logk0"])
        else:
            raise NotImplementedError
        if "k1" in popn_parameters:
            k1 = popn_parameters["k1"]
        elif "logk1" in popn_parameters:
            k1 = np.exp(popn_parameters["logk1"])
        else:
            raise NotImplementedError

        l_hidden = popn_parameters["l_hidden"]
        k_hidden = popn_parameters["k_hidden"]

    else:
        raise NotImplementedError

    # Define the gaussian probability density function in y
    # see doc/20220919_width_of_slow_sequence.txt
    gaussian_y = norm.pdf(y_grid, loc=0, scale=sigma_period)

    # Add the tapering cutoff over a Teff grid using a logistic function.
    # Its midpoint is a function of time.  There are two implemented choices:
    # _teff_0 and teff_zams.  _teff_0 is defined s.t. at 120 Myr it is 4500 K.
    # By 300 Myr it is 4000 K, and at older times it goes below 3800 K.  The
    # alternative teff_zams is more physics-inspired -- at any given time, it
    # is the effective temperature of the lowest-mass star that has just
    # arrived on the ZAMS (as determined using the MIST models).
    teff_logistic_taper = _logistic(
        teff_grid, teff_zams(age, bounds_error=bounds_error), L=1, k=k0
    )

    # N_y X N_teff grid defining the gaussian
    gaussian_y_Teff = gaussian_y[:, None] * teff_logistic_taper[None, :]

    # Slow sequence
    Prot_ss = slow_sequence(
        teff_grid, age, poly_order=poly_order, n=n,
        reference_model_ids=reference_model_ids, reference_ages=reference_ages,
        bounds_error=bounds_error, interp_method=interp_method, verbose=verbose
    ).flatten()

    # Define the uniform component, which we will mold into the time and
    # Teff-evolving fast sequence.
    uniform_y = uniform.pdf(y_grid, loc=y_grid.min(),
                            scale=(y_grid.max()-y_grid.min()))

    # Taper the "upper" part of the uniform ("fast rotator") distribution to
    # avoid overlap with the slow sequence.
    taper_y = 1 - _logistic(y_grid, l_hidden, L=1, k=k_hidden)

    uniform_taper_y = uniform_y * taper_y

    uniform_y_Teff = (
		uniform_taper_y[:, None] * np.ones_like(teff_grid[None, :])
	)

    # Stars must rotate faster than zero days
    below = y_grid[:,None] < -Prot_ss[None,:]
    # Model only the fast sequence -- no positive outliers.
    above = y_grid[:,None] > np.zeros_like(Prot_ss[None,:])

    uniform_y_Teff[(below | above)] = 0

    # Inverse taper for the uniform distribution.  Logic being: stars are
    # moving up from the fast sequence... onto the slow sequence.
    inverse_teff_logistic_taper = 1-teff_logistic_taper

    uniform_y_Teff_0 = uniform_y_Teff * inverse_teff_logistic_taper[None, :]

    # Another inverse logistic taper, same Teff dependence, but with a softer
    # smoothing length.
    uniform_y_Teff_1 = 1.*uniform_y_Teff * (
        1-_logistic(
            teff_grid, teff_zams(age, bounds_error=bounds_error), L=1, k=k1
        )
    )

    a0 = a0
    a1_prefactor = a1 * g_lineardecay(age, bounds_error=bounds_error, y_g=y_g)

    # Initial iteration of model
    resid_y_Teff_0 = a0*gaussian_y_Teff + a1_prefactor*uniform_y_Teff_1

    # marginalize over y_grid
    resid_Teff_0 = np.trapz(resid_y_Teff_0, y_grid, axis=0)

    # normalize to ensure uniform distribution over Teff
    resid_y_Teff = (1/resid_Teff_0[None,:])*(
        a0*gaussian_y_Teff + a1_prefactor*uniform_y_Teff_1
    )

    return resid_y_Teff


def slow_sequence(
    Teff, age, poly_order=7,
    reference_model_ids=[
        'α Per', '120-Myr', '300-Myr', 'Praesepe', 'NGC-6811', '2.6-Gyr', 'M67'
    ],
    reference_ages=[80, 120, 300, 670, 1000, 2600, 4000],
    verbose=True,
    bounds_error='4gyrlimit',
    interp_method='pchip_m67',
    n=None):
    """
    Given an age and a set of temperatures, return the implied slow sequence
    rotation periods, as derived from interpolation using the reference
    clusters with known ages.   This function is the "mean gyrochronal model",
    i.e., it assumes slow sequence evolution.

    Args:

        age (int or float):
            An integer or float corresponding to the age for which we want a
            rotation period.  Units: Myr (=10^6 years).

        Teff (float or iterable of floats):
            Effective temperature(s) of the sample to be dated.  Units: Kelvin.
            Must be between 3800 and 6200 K.

        reference_model_ids (list):
            This list can include any of
            ``['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
            'Praesepe', 'NGC-6811', '120-Myr', '300-Myr', '2.6-Gyr',
            'NGC-6819', 'Ruprecht-147', 'M67']``
            As of gyro-interp v0.4 (i.e., June 2024), the default is set to
            enable gyro-age derivations between 0.08-4 Gyr, and non-physical
            extrapolations past 4 Gyr.  Note that "120-Myr" and "300-Myr" are
            concenations of the relevant clusters.

        reference_ages (iterable of floats):
            Ages (units of Myr) corresponding to ``reference_model_ids``.

        verbose (bool):
            True or False to choose whether to print error messages.  Default
            is False

        interp_method (str):
            How will you interpolate between the polynomial fits to the
            reference open clusters? "pchip_m67" is the suggested default
            method, which uses Piecewise Cubic Hermite Interpolating
            Polynomials (PCHIP) to interpolate over not only 0.8-2.6 Gyr, but
            also sets the gradient in Prot vs Time in the 1-2.6 Gyr interval
            based on the observations of M67 from `Barnes+2016
            <https://ui.adsabs.harvard.edu/abs/2016ApJ...823...16B/abstract>`_,
            `Dungee+2022
            <https://ui.adsabs.harvard.edu/abs/2022ApJ...938..118D/abstract>`_,
            and `Gruner+2023
            <https://ui.adsabs.harvard.edu/abs/2023A%26A...672A.159G/abstract>`_.
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
            different in detail (see the implementation).  This is used only if
            ``interp_method == "alt"`` or ``interp_method == "diff"``, neither
            of which is recommended for most users.  Default is None.

    Output:

        np.ndarray : Prot_slow_sequence

            Numpy array containing period estimate for the slow sequence at the
            given effective temperatures and age, in units of days.
    """

    assert len(reference_ages) == len(reference_model_ids)

    condition0 = (
        interp_method in
        ["skumanich_vary_n", "alt", "diff", "1d_linear", "1d_slinear",
         "1d_quadratic", "1d_pchip", "pchip_m67"]
    )
    condition1 = (
        "skumanich_fix_n" in interp_method
    )
    assert condition0 or condition1

    if not isinstance(Teff, (np.ndarray, list)):
        Teff = np.array([Teff])

    assert age >= 0, "Age must be non-negative."

    # First put everything in age order from youngest reference cluster to
    # oldest reference cluster
    reference_ages = np.array(reference_ages)
    reference_model_ids = np.array(reference_model_ids)
    order = np.argsort(reference_ages)
    reference_ages = reference_ages[order]
    reference_model_ids = reference_model_ids[order]

    # Evaluate each polynomial fit at the given Teff values, to get period
    # estimates for each star in the sample, if it were the age of the
    # reference cluster
    Prot_model_at_known_age = []
    for model_id in reference_model_ids:
        Prot_model = reference_cluster_slow_sequence(
            Teff, model_id, poly_order=poly_order, verbose=False
        )
        Prot_model_at_known_age.append(Prot_model)
    Prot_model_at_known_age = np.array(Prot_model_at_known_age)

    # Special case for pchip_m67
    if interp_method == 'pchip_m67':
        all_reference_model_ids = ['α Per', '120-Myr', '300-Myr', 'Praesepe',
                                   'NGC-6811', '2.6-Gyr', 'M67']
        all_reference_ages = [80, 120, 300, 670, 1000, 2600, 4000]
        all_Prot_model_at_known_age = []
        for model_id in all_reference_model_ids:
            _Prot_model = reference_cluster_slow_sequence(
                Teff, model_id, poly_order=poly_order, verbose=False
            )
            all_Prot_model_at_known_age.append(_Prot_model)
        all_Prot_model_at_known_age = np.array(all_Prot_model_at_known_age)

    # Now start making age estimates
    periods = []

    youngest_model = Prot_model_at_known_age[0]
    oldest_model = Prot_model_at_known_age[-1]

    init_n = deepcopy(n)

    for ix, teff in enumerate(list(Teff)):

        if teff < 3800 or teff > 6200:
            periods.append(np.nan)
            continue

        # special case for if the star has a shorter period than would be expected
        # if it were the age of the youngest reference cluster
        if age < reference_ages[0]:
            if verbose:
                LOGWARNING("Warning! Star is younger than the youngest reference cluster.")
            if bounds_error == 'nan':
                periods.append(np.nan)
            elif bounds_error in ['limit', '4gyrlimit', '4gyrextrap']:
                periods.append(youngest_model[ix])
            else:
                raise NotImplementedError

        # special case for if the star has a longer period than would be expected
        # if it were the age of the oldest reference cluster
        elif (
            age > reference_ages[-1] and
            bounds_error not in ['4gyrlimit', '4gyrextrap']
        ):
            if verbose:
                LOGWARNING("Warning! Star is older than the oldest reference cluster.")
            if bounds_error == 'nan':
                periods.append(np.nan)
            elif bounds_error == 'limit':
                periods.append(oldest_model[ix])
            else:
                raise NotImplementedError

        # special case for when reference age exactly matches desired age
        elif len(reference_ages[age == reference_ages]) > 0:
            sel = (age == reference_ages)
            relevant_cluster = Prot_model_at_known_age[sel,ix]
            periods.append(relevant_cluster)

        else:
            # isolate periods estimates for different ages of star of this mass
            options = Prot_model_at_known_age[:,ix]
            if interp_method == 'pchip_m67':
                all_options = all_Prot_model_at_known_age[:,ix]

            # first identify the youngest cluster older than this star and the
            # oldest cluster younger than this star
            if bounds_error in ["4gyrlimit", "4gyrextrap"] and age >= max(reference_ages):
                # special case: extrapolate based on oldest two clusters
                # needed for interpolation methods that require a rotation
                # period interval: "alt", "diff", and "skumanich_vary_n".
                # for pchip_m67, this is extraneous (if we get to this point
                # and interp_method == "pchip_m67", and bounds_error ==
                # "4gyrlimit", then we will extrapolate using
                # all_reference_model_ids, which includes M67).
                glb_ix = -2
                lub_ix = -1
            else:
                younger, = np.where(reference_ages < age)
                glb_ix = younger[-1]
                older, = np.where(reference_ages > age)
                lub_ix = older[0]

            # 1: older cluster, 0: younger cluster
            P1 = options[lub_ix]
            P0 = options[glb_ix]
            t1 = reference_ages[lub_ix]
            t0 = reference_ages[glb_ix]

            dP = P1 - P0
            dt = t1 - t0

            if interp_method == "alt":
                if n is None:
                    n = 0.5
                C = dP / (t1**n - t0**n)
                fn = lambda age: C * (age**n - t0**n) + P0

            elif interp_method == "diff":
                if n is None:
                    n = 0.5
                C = dP / (dt**n)
                fn = lambda age: C * (age - t0)**n + P0

            elif interp_method == "skumanich_vary_n":
                if init_n is not None:
                    LOGINFO("Over-riding n in interp_method skumanich_vary_n")
                n = np.log(P1/P0) / np.log(t1/t0)
                fn = lambda age: P0 * (age/t0)**n

            elif interp_method in ["1d_linear", "1d_slinear", "1d_quadratic"]:
                kind = interp_method.replace("1d_", "")
                fn = interp1d(reference_ages, options, kind=kind,
                              fill_value="extrapolate")

            elif interp_method == "1d_pchip":
                fn = PchipInterpolator(reference_ages, options)

            elif interp_method == "pchip_m67":
                fn = PchipInterpolator(all_reference_ages, all_options,
                                       extrapolate=True)

            elif "skumanich_fix_n" in interp_method:
                # skumanich_fix_n_{FLOAT_SPECIFYING_N}
                n = float(interp_method.split("_")[-1])
                P0 = options[1] # base off the Pleiades/120myr sequence
                t0 = reference_ages[1]
                fn = lambda age: P0 * (age/t0)**n

            # calculate the period
            period = fn(age)

            # overwrite if necessary based on bounds_error
            if bounds_error == "4gyrlimit" and age > 4000:
                if verbose:
                    LOGINFO("Star is older than the oldest reference cluster...")
                    LOGINFO("\t...You have chosen to not attempt extrapolation.")
                period = fn(4000)
            elif bounds_error == "4gyrextrap" and age > 4000:
                # extrapolate based on local slope
                if verbose:
                    LOGINFO("Star is older than the oldest reference cluster.")
                    LOGINFO("\t...You have chosen to extrapolate based on "
                            "spin-down rate at M67.")
                xmin, xmax = 3990, 4000
                fn2 = interp1d([xmin, xmax], [fn(xmin), fn(xmax)],
                               kind='linear', fill_value='extrapolate')
                period = fn2(age)

            periods.append(period)

    Prot_slow_sequence = np.array(periods)

    if Prot_slow_sequence.shape[-1] == 1:
        Prot_slow_sequence = Prot_slow_sequence.flatten()

    return Prot_slow_sequence


def reference_cluster_slow_sequence(
    Teff, model_id, poly_order=7, verbose=True
    ):
    """
    Given a set of temperatures, get the rotation periods implied by a polymial
    fit to the Prot-Teff slow sequence of a particular cluster between 3800 and
    6200 K.

    Args:

        Teff (np.ndarray / float / list-like iterable):
            Effective temperature in Kelvin.  Curtis+2020 Gaia DR2 BP-RP scale,
            or spectroscopic effective temperatures, preferred above all other
            options.

        model_id (str):
            String identifying the desired reference cluster.  Can be any of
            ``['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
            'Group-X', 'Praesepe', 'NGC-6811', '120-Myr', '300-Myr',
            'NGC-6819', 'Ruprecht-147', '2.6-Gyr', 'M67']``,
            where '120-Myr' will concatenate of Pleiades, Blanco-1, and
            Psc-Eri into one polynomial fit, and '300-Myr' will concatenate
            NGC-3532 and Group-X.

        poly_order (int):
            Integer order of the polynomial fit.

    Returns:

        np.ndarray : Prot_model

            Numpy array containing rotation periods for each requested
            temperature.
    """

    if isinstance(Teff, (list, float, int)):
        Teff = np.array(Teff)

    allowed_model_ids = [
        'α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
        'Praesepe', 'NGC-6811', '120-Myr', '300-Myr', 'NGC-6819',
        'Ruprecht-147', '2.6-Gyr', 'M67'
    ]
    if model_id not in allowed_model_ids:
        raise ValueError(f"Got model_id {model_id} - not implemented!")

    outdir = join(DATADIR, 'interim', 'slow_sequence_coefficients')
    outpath = join(outdir, f'{model_id}_poly{poly_order}_coefficients.txt')

    # Fit the N-th order polynomial to the slow sequence.
    if not os.path.exists(outpath):

        cluster_model_ids = [
            'α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
            'Praesepe', 'NGC-6811', 'NGC-6819', 'Ruprecht-147', 'M67'
        ]
        combined_cluster_ids = {
            '120-Myr': ['Pleiades', 'Blanco-1', 'Psc-Eri'],
            '300-Myr': ['NGC-3532', 'Group-X'],
            '2.6-Gyr': ['NGC-6819', 'Ruprecht-147'],
        }

        cachedir = join(DATADIR, "interim", "slow_sequence_manual_selection")

        if model_id in cluster_model_ids:

            csvpath = join(cachedir, f"{model_id}_slow_sequence.csv")
            assert os.path.exists(csvpath)
            df = pd.read_csv(csvpath)
            _Prot, _Teff = nparr(df['Prot']), nparr(df['Teff_Curtis20'])

        elif model_id in list(combined_cluster_ids.keys()):

            model_ids = combined_cluster_ids[model_id]
            __Prot, __Teff = [], []

            for model_id in model_ids:
                csvpath = join(cachedir, f"{model_id}_slow_sequence.csv")
                assert os.path.exists(csvpath)
                df = pd.read_csv(csvpath)
                __Prot.append(nparr(df['Prot']))
                __Teff.append(nparr(df['Teff_Curtis20']))

            _Prot = np.hstack(__Prot)
            _Teff = np.hstack(__Teff)

        sel = ~pd.isnull(_Prot) & ~pd.isnull(_Teff)
        _Prot, _Teff = _Prot[sel], _Teff[sel]

        coeffs = np.polyfit(_Teff, _Prot, poly_order)

        with open(outpath, 'w') as file_handle:
            np.savetxt(file_handle, coeffs, fmt='%.18e')
        LOGINFO(f"Wrote {outpath}")

    else:
        if verbose:
            LOGINFO(f"Found {outpath}, loading...")

    coeffs = np.genfromtxt(outpath)

    Prot_model = np.polyval(coeffs, Teff)

    # Return NaN for anything below or above allowed temperature range.
    bad_Teff = (Teff > 6200) | (Teff < 3800)
    Prot_model[bad_Teff] = np.nan

    if model_id == 'NGC-6811':
        # Force NGC-6811 to go above Praesepe
        outpath = join(outdir, f'Praesepe_poly{poly_order}_coefficients.txt')
        coeffs_praesepe = np.genfromtxt(outpath)
        Prot_praesepe = np.polyval(coeffs_praesepe, Teff)
        below_praesepe = Prot_model < Prot_praesepe
        Prot_model[below_praesepe] = Prot_praesepe[below_praesepe] + 0.01

    return Prot_model
