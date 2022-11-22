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

def given_grid_post_get_summary_statistics(age_grid, age_post, N=int(1e5)):

    age_peak = int(age_grid[np.argmax(age_post)])

    df = pd.DataFrame({'age':age_grid, 'p':age_post})
    sample_df = df.sample(n=N, replace=True, weights=df.p)

    one_sig = 68.27/2
    two_sig = 95.45/2
    three_sig = 99.73/2

    pct_50 = np.nanpercentile(sample_df.age, 50)

    p1sig = np.nanpercentile(sample_df.age, 50+one_sig) - pct_50
    m1sig = pct_50 - np.nanpercentile(sample_df.age, 50-one_sig)

    p2sig = np.nanpercentile(sample_df.age, 50+two_sig) - pct_50
    m2sig = pct_50 - np.nanpercentile(sample_df.age, 50-two_sig)

    p3sig = np.nanpercentile(sample_df.age, 50+three_sig) - pct_50
    m3sig = pct_50 - np.nanpercentile(sample_df.age, 50-three_sig)

    outdict = {
        'median': np.round(pct_50,2),
        'peak': np.round(age_peak,2),
        'mean': np.round(np.nanmean(sample_df.age),2),
        '+1sigma': np.round(p1sig,2),
        '-1sigma': np.round(m1sig,2),
        '+2sigma': np.round(p2sig,2),
        '-2sigma': np.round(m2sig,2),
        '+3sigma': np.round(p3sig,2),
        '-3sigma': np.round(m3sig,2),
        '+1sigmapct': np.round(p1sig/pct_50,2),
        '-1sigmapct': np.round(m1sig/pct_50,2),
    }

    return outdict


def get_age_posterior_worker(task):

    Prot, Teff, age_grid, outdir = task

    Protstr = f"{float(Prot):.2f}"
    Teffstr = f"{float(Teff):.1f}"
    typestr = 'limitgrid'
    bounds_error = 'limit'

    cachepath = os.path.join(outdir, f"Prot{Protstr}_Teff{Teffstr}_{typestr}.csv")
    if not os.path.exists(cachepath):
        age_post = gyro_age_posterior(
            Prot, Teff, age_grid=age_grid, bounds_error=bounds_error,
            verbose=False
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


def main():

    outdir = os.path.join(LOCALDIR, "gyroemp")
    if not os.path.exists(outdir): os.mkdir(outdir)
    outdir = os.path.join(LOCALDIR, "gyroemp", "prot_teff_grid")
    if not os.path.exists(outdir): os.mkdir(outdir)

    age_grid = np.linspace(0, 2600, 500)
    teffmin, teffmax = 3800, 6200
    protmin, protmax = 0, 23
    Teff_grid = np.arange(teffmin, teffmax+50, 50)
    Prot_grid = np.arange(protmin, protmax+0.5, 0.5)

    tasks = [(_prot, _teff, age_grid, outdir) for _prot, _teff in
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
    main()
