import os
import numpy as np
import gyrointerp.plotting as gp
from gyrointerp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'extrapolation_test_prot_vs_time_fixed_teff')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

for xscale in ['linear', 'log']:
    for teff in [4199, 4999, 5799]:
        gp.plot_prot_vs_time_fixed_teff(
            PLOTDIR, teff, ['pchip_m67', 'pchip_m67'], xscale=xscale,
            bounds_errors=['4gyrlimit', '4gyrextrap'],
            ages=np.linspace(50, 10000, 300)
        )
