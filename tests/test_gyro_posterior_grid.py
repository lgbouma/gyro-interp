import os, pytest
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrointerp.gyro_posterior import (
    gyro_age_posterior, _agethreaded_gyro_age_posterior
)
from aesthetic.plot import set_style
from datetime import datetime

@pytest.mark.skip(reason="setting up CI")
def test_gyro_posterior_Ngrid(Teff=5500, Teff_err=50, Prot=10, Prot_err=0.1):

    plt.close("all")
    set_style("clean")
    fig, ax = plt.subplots()

    for N_grid in [128, 256, 512, 1024]:

        age_grid, age_post = run(
            Prot, Teff, N_grid, Prot_err=Prot_err, Teff_err=Teff_err
        )

        ax.plot(age_grid, 1e3*age_post, label='N$_\mathrm{grid}=$'+f'{N_grid}', lw=0.5)

    ax.legend(loc='best', fontsize='x-small')
    ax.set_title(f'Teff {Teff}, Prot {Prot}, Proterr {Prot_err}, 300 age points')
    ax.set_xlabel("Age [Myr]")
    ax.set_ylabel("Probability [$10^{-3}\,$Myr$^{-1}$]")
    outname = f"grid_size_check_Teff{Teff}_Prot{Prot}_Proterr{Prot_err}.png"
    outpath = os.path.join('../results/debug/', outname)
    plt.savefig(outpath, dpi=400)


@pytest.mark.skip(reason="helper function")
def run(Prot, Teff, N, Prot_err=None, Teff_err=None):

    bounds_error = 'limit'
    age_grid = np.linspace(0, 2700, 301)

    print(f"{datetime.now().isoformat()} begin N={N}")
    age_post = gyro_age_posterior(
        Prot, Teff, Teff_err=Teff_err, Prot_err=Prot_err, age_grid=age_grid,
        bounds_error=bounds_error, N_grid=N, verbose=False
    )
    print(f"{datetime.now().isoformat()} end N={N}")
    return age_grid, age_post


if __name__ == "__main__":
    test_gyro_posterior_Ngrid(Teff=5500, Prot=10, Prot_err=0.1)
    test_gyro_posterior_Ngrid(Teff=5500, Prot=10, Prot_err=0.01)
    test_gyro_posterior_Ngrid(Teff=3810, Prot=1, Prot_err=0.1)
    test_gyro_posterior_Ngrid(Teff=3810, Prot=1, Prot_err=0.01)
