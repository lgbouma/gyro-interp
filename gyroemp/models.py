"""
Functions to fit rotation-Teff sequences, or to quickly return the results of
those fits (including their interpolations!)

Contents:
    reference_cluster_slow_sequence
    slow_sequence
    slow_sequence_residual

Helper functions:
    logistic
    teff_zams
    teff_0
    C_uniform
"""
import os, pickle
from gyroemp.paths import DATADIR
import pandas as pd, numpy as np
from numpy import array as nparr
from os.path import join

###########
# helpers #
###########
def logistic(x, x0, L=1, k=0.1):
    # larger k makes the cutoff sharper
    num = L
    denom = 1 + np.exp(-k * (x-x0))
    return num / denom


def teff_zams(age, bounds_error='limit'):
    """
    Physics-informed MIST effective temperature for the effective temperature a
    star has when it arrives on the ZAMS.

    Defined for age from 80 to 1000 Myr.  If `bounds_error=='limit'`, then set
    as whatever the lowest and highest values are.

    Tested at /tests/plot_teff_cuts.py
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
    teff0 = 10**spl(np.log10(age*1e6))

    max_teff = 10**spl(np.log10(80*1e6))
    min_teff = 10**spl(np.log10(1e9))

    bad = (age < 80) | (age > 1000)
    if bounds_error == 'nan':
        teff0[bad] = np.nan
    elif bounds_error == 'limit':
        teff0[age < 80] = max_teff
        teff0[age > 1000] = min_teff

    return teff0


def teff_0(age, bounds_error='limit'):
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
    elif bounds_error == 'limit':
        teff0[age < 120] = c
        teff0[age > 1000] = -(1000-120) * slope + c

    return teff0


def C_uniform(age, bounds_error='limit', y0=1/2):
    """
    How the third uniform component amplitude from slow_sequence_residual
    decreases with age Full amplitude at 120 Myr.  `y0` amplitude by 300 Myr
    (eg. 1/2, 1/4, 1/6).
    Decreasing linearly thereafter, and floor at zero.  Defined for age from
    120 to 1000 Myr.
    """

    if isinstance(age, (float,int)):
        age = np.array([age])

    # units: 1/Myr
    y1 = 1
    slope = (y1 - y0) / (300 - 120)
    c = y1*1.

    c_uniform = -(age-120) * slope + c

    bad = (age < 120) | (age > 1000)

    if bounds_error == 'nan':
        c_uniform[bad] = np.nan
    elif bounds_error == 'limit':
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
    poly_order=7, n=0.5,
    reference_model_ids=[
        'α Per', '120-Myr', '300-Myr', 'Praesepe', 'NGC-6811', '2.6-Gyr'
    ],
    reference_ages=[80, 120, 300, 670, 1000, 2600],
    popn_parameters='default',
    verbose=True,
    bounds_error='limit'):
    """
    Given an effective temperature and an age, return the 2-D distribution of
    residuals around and underneath the slow sequence, sampled onto grids of
    "y_grid" X "teff_grid", where `y_grid` is the residual of (rotation period
    data - model).

    This model for the residual has a choice between two or three components:

        * a gaussian in "y_grid" , with an age-varying cutoff in Teff, imposed
        as a logistic taper.

        * (optional) an age-varying and Teff-varying uniform distribution,
        multiplied by the inverse the gaussian's taper, and then truncated to
        ensure that stars rotate faster than zero days, and to ensure that we
        model only the fast sequence.  This uniform distribution is also
        tapered by a logistic function at the "slow" end to yield a smoother
        transition to the gaussian.

        * an age-varying and Teff-invariant uniform distribution, similar to
        the one described above, that accounts for the few fast-rotating
        outliers by having a longer taper scale length.

    Args:
        y_grid, teff_grid, poly_order, reference_model_ids, reference_ages:
            self-explanatory, except for `y_grid`, which was defined further up
            in the docstring.

        bounds_error: "nan" or "limit".  If "nan" then below the minimum
        reference age, returns nans.  If "limit", then return the limiting
        residual distribution (usually that of the 120-Myr clusters, when below
        their sequence).

        popn_parameters: (str) "default", or (dict) containing the
        population-level free parameters.  Keys of "A", "C", "C_y0", "k0",
        "l1", "k1", and "k2" must all be specified.  If "B" is not specified,
        assumes B=0, and the optional component mentioned above is omitted.

    Returns:
        resid_y_Teff: 2d array with dimension (N_y_grid x N_teff_grid)
    """

    assert len(reference_ages) == len(reference_model_ids)

    from scipy.stats import norm, uniform

    # The intrinsic width (RMS) of the slow sequence, in units of days.
    sigma_period = 0.51

    if popn_parameters == "default":
        # from run_emcee_fit_gyro_model; MAP-values
        # [8.25637486, 0.6727635, -4.8845869, -6.23968718, -0.14829162]
        A = 1
        B = 0
        C = 8.256
        C_y0 = 0.673
        k0 = np.e**-4.885
        l1 = -2*sigma_period
        k1 = np.pi # a joke, but it works
        k2 = np.e**-6.240


    elif isinstance(popn_parameters, dict):

        A = popn_parameters["A"]

        if "B" in popn_parameters:
            B = popn_parameters["B"]
        elif "logB" in popn_parameters:
            B = np.exp(popn_parameters["logB"])
        else:
            if verbose:
                print("B not explicitly set; assuming fully omitted.")
            B = 0

        C = popn_parameters["C"]
        C_y0 = popn_parameters["C_y0"]

        if "k0" in popn_parameters:
            k0 = popn_parameters["k0"]
        elif "logk0" in popn_parameters:
            k0 = np.exp(popn_parameters["logk0"])
        else:
            raise NotImplementedError

        l1 = popn_parameters["l1"]
        k1 = popn_parameters["k1"]

        if "k2" in popn_parameters:
            k2 = popn_parameters["k2"]
        elif "logk2" in popn_parameters:
            k2 = np.exp(popn_parameters["logk2"])
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    # Define the gaussian probability density function in y
    # see doc/20220919_width_of_slow_sequence.txt
    gaussian_y = norm.pdf(y_grid, loc=0, scale=sigma_period)

    # Add the tapering cutoff over a Teff grid using a logistic function.
    # Its midpoint is a function of time.  There are two implemented choices:
    # teff_0 and teff_zams.  teff_0 is defined s.t. at 120 Myr it is 4500 K.
    # By 300 Myr it is 4000 K, and at older times it goes below 3800 K.  The
    # alternative teff_zams is more physics-inspired -- at any given time, it
    # is the effective temperature of the lowest-mass star that has just
    # arrived on the ZAMS (as determined using the MIST models).
    teff_logistic_taper = logistic(
        teff_grid, teff_zams(age, bounds_error=bounds_error), L=1, k=k0
    )

    # N_y X N_teff grid defining the gaussian
    gaussian_y_Teff = gaussian_y[:, None] * teff_logistic_taper[None, :]

    # Slow sequence
    Prot_ss = slow_sequence(
        teff_grid, age, poly_order=poly_order, n=n,
        reference_model_ids=reference_model_ids, reference_ages=reference_ages,
        bounds_error=bounds_error, verbose=verbose
    ).flatten()

    # Define the uniform component, which we will mold into the time and
    # Teff-evolving fast sequence.
    uniform_y = uniform.pdf(y_grid, loc=y_grid.min(),
                            scale=(y_grid.max()-y_grid.min()))

    # Taper the "upper" part of the uniform ("fast rotator") distribution to
    # avoid overlap with the slow sequence.
    taper_y = 1 - logistic(y_grid, l1, L=1, k=k1)

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
        1-logistic(
            teff_grid, teff_zams(age, bounds_error=bounds_error), L=1, k=k2
        )
    )

    A = A
    B = B
    C_prefactor = C * C_uniform(age, bounds_error=bounds_error, y0=C_y0)

    # Initial iteration of model
    resid_y_Teff_0 = A*gaussian_y_Teff + B*uniform_y_Teff_0 + C_prefactor*uniform_y_Teff_1

    # marginalize over y_grid
    resid_Teff_0 = np.trapz(resid_y_Teff_0, y_grid, axis=0)

    # normalize to ensure uniform distribution over Teff
    resid_y_Teff = (1/resid_Teff_0[None,:])*(
        A*gaussian_y_Teff + B*uniform_y_Teff_0 + C_prefactor*uniform_y_Teff_1
    )

    return resid_y_Teff


def slow_sequence(
    Teff, age, poly_order=7, n=0.5,
    reference_model_ids=[
        'α Per', '120-Myr', '300-Myr', 'Praesepe', 'NGC-6811', '2.6-Gyr'
    ],
    reference_ages=[80, 120, 300, 670, 1000, 2600],
    verbose=True,
    bounds_error='limit'):
    """
    Predicts a star's rotation period based on its age and those of reference
    clusters with known ages. Assumes that the star is on the slow sequence and
    that it follows a power-law spin-down.

    Age must be between the lowest and highest reference age, otherwise an
    error will be raised.  Teff must be between 3800 and 6200 K.

    Args:

        age: an integer or float corresponding to the age that we want a
        rotational period for.  units: Myr

        Teff: float or iterable of floats corresponding to the effective
        temperature(s) of the sample to be dated.  units: K

        reference_model_ids: list including any of
            ['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
            'Praesepe', 'NGC-6811', '120-Myr', '300-Myr', '2.6-Gyr',
            'NGC-6819', 'Ruprecht-147']
        The sensible default is set.  120-Myr and 300-Myr are concenations of
        the relevant clusters.

        reference_ages: iterable of ages corresponding to reference_model_ids.

        n: braking index, defined by the spin-down power law Prot ~ t^n. Default
        is the canonical 0.5

        verbose: input True or False to choose whether to print error messages.
        Default is False

        bounds_error: "nan" or "limit".  If "nan" ages below the minimum
        reference age return nans.  If "limit", they return the limiting
        rotation period at the closest cluster.  Default "limit".

    Output:

        Numpy array containing period estimate at the given effective
        temperatures and age, in the same units as the provided reference
        periods.
    """

    assert len(reference_ages) == len(reference_model_ids)

    if not isinstance(Teff, (np.ndarray, list)):
        Teff = np.array([Teff])

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

    # Now start making age estimates
    periods = []

    youngest_model = Prot_model_at_known_age[0]
    oldest_model = Prot_model_at_known_age[-1]

    for ix, teff in enumerate(list(Teff)):

        # special case for if the star has a shorter period than would be expected
        # if it were the age of the youngest reference cluster
        if age < reference_ages[0]:
            if verbose:
                print("Warning! Star is younger than the youngest reference cluster.")
            if bounds_error == 'nan':
                periods.append(np.nan)
            if bounds_error == 'limit':
                if verbose:
                    print("Taking youngest reference cluster Prot as the upper limit.")
                periods.append(youngest_model[ix])

        # special case for if the star has a longer period than would be expected
        # if it were the age of the oldest reference cluster
        elif age > reference_ages[-1]:
            if verbose:
                print("Warning! Star is older than the oldest reference cluster.")
            if bounds_error == 'nan':
                periods.append(np.nan)
            if bounds_error == 'limit':
                if verbose:
                    print("Taking oldest reference cluster Prot as the lower limit.")
                periods.append(oldest_model[ix])

        # special case for when reference age exactly matches desired age
        elif len(reference_ages[age == reference_ages]) > 0:
            sel = (age == reference_ages)
            relevant_cluster = Prot_model_at_known_age[sel,ix]
            periods.append(relevant_cluster)

        else:
            # isolate periods estimates for different ages of star of this mass
            options = Prot_model_at_known_age[:,ix]

            # first identify the youngest cluster older than this star and the
            # oldest cluster younger than this star
            younger, = np.where(reference_ages < age)

            # in order from youngest to oldest so choose last index for oldest
            # younger than this star
            glb_ix = younger[-1]

            older, = np.where(reference_ages > age)

            lub_ix = older[0]

            dP = options[lub_ix] - options[glb_ix]
            # older one
            tg1 = reference_ages[lub_ix]
            # younger one
            tg0 = reference_ages[glb_ix]

            C = dP / (tg1**n - tg0**n)

            Prot_glb = options[glb_ix]

            period = C*(age**n - tg0**n) + Prot_glb

            periods.append(period)

    return_arr = np.array(periods)

    if return_arr.shape[-1] == 1:
        return_arr = return_arr.flatten()

    return return_arr


def reference_cluster_slow_sequence(
    Teff, model_id, poly_order=7, verbose=True
    ):
    """
    Given Teff, get Prot implied by a polymial fit to the Prot-Teff slow
    sequence of a given cluster between 3800 and 6200 K.  Note -- this means
    the upper envelope.

    Args:

        Teff (np.ndarray / float / list-like iterable): effective temperature
        in Kelvin.  Curtis+2020 Gaia DR2 BP-RP scale, or spectroscopic
        effective temperatures, preferred above all other options.

        model_id (str): any of
            ['α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
            'Praesepe', 'NGC-6811', '120-Myr', '300-Myr']
        where '120-Myr' will concatenate of Pleiades, Blanco-1, and
        Psc-Eri, and '300-Myr' will concatenate NGC-3532 and Group-X.

        poly_order (int): integer for the polynomial fit.
    """

    if isinstance(Teff, (list, float, int)):
        Teff = np.array(Teff)

    allowed_model_ids = [
        'α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
        'Praesepe', 'NGC-6811', '120-Myr', '300-Myr', 'NGC-6819',
        'Ruprecht-147', '2.6-Gyr'
    ]
    if model_id not in allowed_model_ids:
        raise ValueError(f"Got model_id {model_id} - not implemented!")

    outdir = join(DATADIR, 'interim', 'slow_sequence_coefficients')
    outpath = join(outdir, f'{model_id}_poly{poly_order}_coefficients.txt')

    # Fit the N-th order polynomial to the slow sequence.
    if not os.path.exists(outpath):

        cluster_model_ids = [
            'α Per', 'Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532', 'Group-X',
            'Praesepe', 'NGC-6811', 'NGC-6819', 'Ruprecht-147'
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
        print(f"Wrote {outpath}")

    else:
        if verbose:
            print(f"Found {outpath}, loading...")

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


if __name__ == "__main__":
    # debugging
    _ = reference_cluster_slow_sequence([4321], 'Pleiades')
    print(_)
    _ = reference_cluster_slow_sequence([4321], '120-Myr')
    print(_)
