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

from gyroemp.gyro_posterior import gyro_age_posterior
from gyroemp.paths import RESULTSDIR, LOCALDIR
from gyroemp.helpers import given_grid_post_get_summary_statistics

def get_age_posterior_worker(task):

    Prot, Teff, age_grid, outdir, n = task

    Protstr = f"{float(Prot):.2f}"
    Teffstr = f"{float(Teff):.1f}"
    typestr = 'limitgrid'
    bounds_error = 'limit'

    cachepath = os.path.join(outdir, f"Prot{Protstr}_Teff{Teffstr}_{typestr}.csv")
    if not os.path.exists(cachepath):
        age_post = gyro_age_posterior(
            Prot, Teff, age_grid=age_grid, bounds_error=bounds_error,
            verbose=False, n=n
        )
        df = pd.DataFrame({
            'age_grid': age_grid,
            'age_post': age_post
        })
        outpath = cachepath.replace(".csv", "_posterior.csv")
        df.to_csv(outpath, index=False)
        print(f"Wrote {outpath}")

        d = given_grid_post_get_summary_statistics(age_grid, age_post)
        d['Prot'] = Prot
        d['Teff'] = Teff
        df = pd.DataFrame(d, index=[0])
        df.to_csv(cachepath, index=False)
        print(f"Wrote {cachepath}")

        return 1

    else:
        print(f"Found {cachepath}")
        return 1


def main(n=1.0):

    outdir = os.path.join(LOCALDIR, "gyroemp")
    if not os.path.exists(outdir): os.mkdir(outdir)

    outdir = os.path.join(
        LOCALDIR, "gyroemp", f"prot_teff_grid_n{n}_reluncpt1pct"
    )
    if not os.path.exists(outdir): os.mkdir(outdir)

    age_grid = np.linspace(0, 2600, 500)
    teffmin, teffmax = 3800, 6200
    protmin, protmax = 0, 23
    Teff_grid = np.arange(teffmin, teffmax+50, 50)
    Prot_grid = np.arange(protmin, protmax+0.5, 0.5)

    tasks = [(_prot, _teff, age_grid, outdir, n) for _prot, _teff in
             product(Prot_grid, Teff_grid)]

    N_tasks = len(tasks)
    print(f"Got N_tasks={N_tasks}...")
    print(f"{datetime.now().isoformat()} begin")

    nworkers = mp.cpu_count()
    maxworkertasks= 1000

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    results = pool.map(get_age_posterior_worker, tasks)

    pool.close()
    pool.join()

    print(f"{datetime.now().isoformat()} end")


if __name__ == "__main__":
    main(n=1.0)
    main(n=0.2)
    main(n=0.5)
