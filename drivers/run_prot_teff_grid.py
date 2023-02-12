"""
Define a grid of Teff and Prot over which to compute the gyro age posteriors.
Parallelize over stars, and get linear speedup with the number of cores on your
machine.

(on wh1, this yielded 220 posteriors in 40 seconds)
"""
import os
from datetime import datetime
import multiprocessing as mp
import numpy as np, pandas as pd
from itertools import product

from gyrointerp.gyro_posterior import (
    gyro_age_posterior, _one_star_age_posterior_worker
)
from gyrointerp.paths import RESULTSDIR, LOCALDIR
from gyrointerp.helpers import get_summary_statistics

def main(n=None, age_scale="default", interp_method="pchip_m67",
         grid_resolution="coarse"):
    """
    n: assume Prot ~ t^{n} scaling

    age_scale: "default", "1sigmaolder", or "1sigmayounger".  Shifts the
    entire age scale appropriately.
    """

    outdir = os.path.join(LOCALDIR, "gyrointerp")
    if not os.path.exists(outdir): os.mkdir(outdir)

    outdir = os.path.join(
        LOCALDIR, "gyrointerp",
        f"prot_teff_grid_n{n}_reluncpt1pct_{age_scale}_{interp_method}"
    )
    if not os.path.exists(outdir): os.mkdir(outdir)

    age_grid = np.linspace(0, 2600, 500)
    teffmin, teffmax = 3800, 6200
    protmin, protmax = 0, 23.5
    if grid_resolution == "fine":
        Teff_grid = np.arange(teffmin, teffmax+10, 10)
        Prot_grid = np.arange(protmin, protmax+0.1, 0.1)
    elif grid_resolution == "coarse":
        Teff_grid = np.arange(teffmin, teffmax+100, 100)
        Prot_grid = np.arange(protmin, protmax+0.5, 0.5)

    bounds_error = "4gyrlimit"
    parameters = 'default'
    N_grid = 'default'

    tasks = [
        (_prot, _teff, None, None, None, age_grid, outdir, bounds_error,
         interp_method, n, age_scale, parameters, N_grid)
         for _prot, _teff in product(Prot_grid, Teff_grid)
    ]

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
    # Run required for figure 3 (the precision figure)
    main(n=None, age_scale="default", interp_method="pchip_m67",
         grid_resolution="fine")
    assert 0

    # Systematic shift based on interpolation method / time-dependent spin-down
    main(n=None, age_scale="default", interp_method="skumanich_vary_n")

    main(n=None, age_scale="default", interp_method="1d_linear")
    main(n=None, age_scale="default", interp_method="1d_pchip")
    main(n=None, age_scale="default", interp_method="pchip_m67")
    assert 0


    # Systematic shift based on age scale
    main(n=None, age_scale="default")
    main(n=None, age_scale="1sigmaolder")
    main(n=None, age_scale="1sigmayounger")

    # # DEPRECATED
    # main(n=1.0)
    # main(n=0.2)
    # main(n=0.5)
