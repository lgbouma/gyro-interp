import os
import gyrointerp.plotting as gp
from gyrointerp.paths import RESULTSDIR
import numpy as np, pandas as pd

age_grid = np.linspace(0, 2600, 1000)
Teffs_Prots = [
    (5800, np.arange(2, 12, 1.5)),
    (5000, np.arange(4, 13.5, 1.5)),
    (4200, np.arange(9, 18, 1.5)),
    #(4000, np.arange(10, 12, 1.5)),
    #(4000, np.arange(10, 18, 1.5)),
    #(4000, np.arange(9.5, 18, 1.5)),
    #(6000, np.arange(2,10,1.5)),
    #(5500, np.arange(3,13,1.5)),
    #(4500, np.arange(6, 16, 1.5)),
    #(5500, np.arange(3.5,13,1.5)),
    #(4500, np.arange(6.5, 16, 1.5))
]

#
# complicated test, with population level sampling MCMC
#
PLOTDIR = os.path.join(RESULTSDIR, 'multi_age_posteriors_mcmc')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR
for t in Teffs_Prots:
    Teff, Prots = t
    _outdir = os.path.join(
        outdir, f"Teff{Teff}_Prots{'_'.join(Prots.astype(str))}"
    )
    if not os.path.exists(_outdir): os.mkdir(_outdir)
    gp.plot_age_posteriors(
        Prots, Teff, _outdir, age_grid=age_grid, full_mcmc=1
    )
assert 0


#
# simple default, no population level sampling MCMC
#
PLOTDIR = os.path.join(RESULTSDIR, 'multi_age_posteriors')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR
for t in Teffs_Prots:
    Teff, Prots = t
    gp.plot_age_posteriors(
        Prots, Teff, outdir, age_grid=age_grid
    )


