import os
import gyroemp.plotting as ap
from gyroemp.paths import RESULTSDIR
import numpy as np, pandas as pd

PLOTDIR = os.path.join(RESULTSDIR, 'multi_age_posteriors')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
outdir = PLOTDIR

age_grid = np.linspace(0, 2600, 500)

Teff = 4000
Prots = np.arange(10, 18, 1.5)
ap.plot_age_posteriors(
    Prots, Teff, outdir, age_grid=age_grid
)

Teff = 4000
Prots = np.arange(9.5, 18, 1.5)
ap.plot_age_posteriors(
    Prots, Teff, outdir, age_grid=age_grid
)

Teff = 6000
Prots = np.arange(2,10,1.5)
ap.plot_age_posteriors(
    Prots, Teff, outdir, age_grid=age_grid
)

Teff = 5500
Prots = np.arange(3,13,1.5)
ap.plot_age_posteriors(
    Prots, Teff, outdir, age_grid=age_grid
)

Teff = 4500
Prots = np.arange(6, 16, 1.5)
ap.plot_age_posteriors(
    Prots, Teff, outdir, age_grid=age_grid
)

Teff = 5500
Prots = np.arange(3.5,13,1.5)
ap.plot_age_posteriors(
    Prots, Teff, outdir, age_grid=age_grid
)

Teff = 4500
Prots = np.arange(6.5, 16, 1.5)
ap.plot_age_posteriors(
    Prots, Teff, outdir, age_grid=age_grid
)
