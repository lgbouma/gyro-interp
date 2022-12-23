"""
(deprecated)
"""
import os
import gyrointerp.plotting as ap
from gyrointerp.paths import RESULTSDIR
import numpy as np, pandas as pd

PLOTDIR = os.path.join(RESULTSDIR, 'age_posteriors_boundserror_limit')
if not os.path.exists(PLOTDIR): os.mkdir(PLOTDIR)
outdir = PLOTDIR

bounds_error = 'limit'
age_grid = np.linspace(0, 2600, 500)

for Teff in [4500, 5500]:
    for Prot in np.arange(5,15):
        ap.plot_age_posterior(
            Prot, Teff, outdir, age_grid=age_grid, bounds_error=bounds_error
        )

