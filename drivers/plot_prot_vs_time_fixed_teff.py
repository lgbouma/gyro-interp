import os
import gyrointerp.plotting as gp
from gyrointerp.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'prot_vs_time_fixed_teff')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

teffs = [5800, 5000, 4200]
n = [0.47, 0.33, 0.25]

for teff, _n in zip(teffs, n):

    interp_methods = [
        f"skumanich_fix_n_{_n}",
        "skumanich_vary_n", # "alt", "diff", #"skumanich_fix_n_0.5",
        "1d_linear", "1d_quadratic", #"1d_slinear",
        "1d_pchip",
        "pchip_m67"
    ]

    gp.plot_prot_vs_time_fixed_teff(PLOTDIR, teff, interp_methods,
                                    xscale='log')
    gp.plot_prot_vs_time_fixed_teff(PLOTDIR, teff, interp_methods,
                                    xscale='linear')
