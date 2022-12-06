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

from gyroemp.gyro_posterior import (
    gyro_age_posterior, _one_star_age_posterior_worker
)
from gyroemp.paths import RESULTSDIR, LOCALDIR
from gyroemp.helpers import given_grid_post_get_summary_statistics

def main(n=0.5, age_scale="default"):
    """
    n: assume Prot ~ t^{n} scaling

    age_scale: "default", "1sigmaolder", or "1sigmayounger".  Shifts the
    entire age scale appropriately.
    """

    outdir = os.path.join(LOCALDIR, "gyroemp")
    if not os.path.exists(outdir): os.mkdir(outdir)

    outdir = os.path.join(
        LOCALDIR, "gyroemp", f"prot_teff_grid_n{n:.1f}_reluncpt1pct_{age_scale}"
    )
    if not os.path.exists(outdir): os.mkdir(outdir)

    age_grid = np.linspace(0, 2600, 500)
    teffmin, teffmax = 3800, 6200
    protmin, protmax = 0, 23
    Teff_grid = np.arange(teffmin, teffmax+50, 50)
    Prot_grid = np.arange(protmin, protmax+0.5, 0.5)

    tasks = [(_prot, _teff, age_grid, outdir, n, age_scale, 'default')
             for _prot, _teff in product(Prot_grid, Teff_grid)]

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
    main(n=0.5, age_scale="default")
    main(n=0.5, age_scale="1sigmaolder")
    main(n=0.5, age_scale="1sigmayounger")
    assert 0
    main(n=1.0)
    main(n=0.2)
    main(n=0.5)
