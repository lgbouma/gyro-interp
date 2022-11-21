import os
import multiprocessing as mp
from gyroemp.paths import RESULTSDIR, LOCALDIR
import numpy as np, pandas as pd
from itertools import product

outdir = os.path.join(LOCALDIR, "gyroemp")
if not os.path.exists(outdir): os.mkdir(outdir)

def get_age_posterior_worker(task):

    Prot, Teff, age_grid = task

    Protstr = f"{Prot:.2f}"
    Teffstr = f"{Teff:.1f}"
    typestr = 'limitgrid'
    bounds_error = 'limit'

    cachepath = os.path.join(outdir, f"Prot{Protstr}_Teff{Teff}_{typestr}.csv")
    if not os.path.exists(cachepath):
        age_post = gyro_age_posterior(
            Prot, Teff, age_grid=age_grid, bounds_error=bounds_error,
            verbose=False
        )
        df = pd.DataFrame({
            'age_grid': age_grid,
            'age_post': age_post
        })
        df.to_csv(cachepath)
        print(f"Wrote {cachepath}")
        return 1
    else:
        print(f"Found {cachepath}")
        return 1



def main():

    age_grid = np.linspace(0, 2600, 500)
    Teff_grid = np.arange(3800, 6200+200, 200)
    Prot_grid = np.arange(0, 17, 1)

    tasks = [(_teff, _prot, age_grid) for _teff, _prot in
             product(Teff_grid, Prot_grid)]

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

    import IPython; IPython.embed()
    #FIXME


if __name__ == "__main__":
    main()
