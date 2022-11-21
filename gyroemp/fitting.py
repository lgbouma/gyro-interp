import numpy as np, pandas as pd
import os
from gyroemp.paths import RESULTSDIR
from gyroemp.plotting import _get_model_histogram

def get_chi_sq_red(parameters, verbose=1):

    model_ids = ['115-Myr', '300-Myr', 'Praesepe']
    reference_clusters = ['Pleiades', 'Blanco-1', 'Psc-Eri', 'NGC-3532',
                          'Group-X', 'Praesepe', 'NGC-6811']

    ages = [115, 300, 670]
    cachedir = os.path.join(RESULTSDIR, 'cdf_fast_slow_ratio')
    model_ids = ['115-Myr', '300-Myr', 'Praesepe']

    chi_sqs = []
    for age, model_id in zip(ages, model_ids):

        # Get the data
        csvpath = os.path.join(RESULTSDIR, 'cdf_fast_slow_ratio',
                               f'{model_id}_cdf_fast_slow_ratio_data.csv')
        df = pd.read_csv(csvpath)

        data_midpoints = np.array(df.Teff_midpoints)
        data_ratio = np.array(df.ratio)

        h_vals_ss, h_vals_fs, teff_midway = _get_model_histogram(
            age, parameters=parameters
        )

        model_midpoints = teff_midway
        model_ratio = np.array( h_vals_fs / (h_vals_fs + h_vals_ss) )

        if age in [115, 300]:
            sigma = 0.1 # uniform weighting across the 7 bins
        elif age == 670:
            sigma = 0.01 # stricter requirement -- want it gonezo.
        else:
            raise NotImplementedError

        chi_sq = np.sum( (data_ratio - model_ratio)**2 / sigma**2 )
        chi_sqs.append(chi_sq)

    n = 21
    k = 6
    if 'B' in parameters:
        if parameters['B'] == 0:
            k = 5
    chi_sq = np.sum(chi_sqs)
    chi_sq_red = chi_sq / (n-k)
    BIC = chi_sq + k*np.log(n)

    if verbose:
        print(parameters)
        print(f"this model χ^2_red: {chi_sq_red:.4f}, χ^2: {chi_sq:.1f}, BIC: {BIC:.2f}")

    return chi_sq_red, BIC
