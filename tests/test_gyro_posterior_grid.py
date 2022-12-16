import os, pytest
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrointerp.gyro_posterior import gyro_age_posterior
from datetime import datetime

@pytest.mark.skip(reason="setting up CI")
def test_gyro_posterior_grid_sunlike():

    bounds_error = 'limit'
    age_grid = np.linspace(0, 2700, 101)
    Teff = 5500
    Prot = 10

    def run(N):

        print(f"{datetime.now().isoformat()} begin N={N}")
        age_post = gyro_age_posterior(
            Prot, Teff, age_grid=age_grid, bounds_error=bounds_error,
            N_grid=N, verbose=False
        )
        print(f"{datetime.now().isoformat()} end N={N}")
        return age_post

    plt.close("all")
    plt.plot(age_grid, run(64), label='N=64', lw=0.5)
    plt.plot(age_grid, run(128), label='N=128', lw=0.5)
    plt.plot(age_grid, run(256), label='N=256', lw=3)
    plt.plot(age_grid, run(512), label='N=512', lw=0.5)
    #TODO FIXME: formalize criterion for assessing "convergence" here.
    plt.legend(loc='best', fontsize='x-small')
    plt.title('teff 5500, prot 10 days, 100 age points')
    plt.xlabel("age")
    plt.ylabel("probability")
    plt.savefig('../results/debug/grid_size_check_sunlike.png', dpi=300)


@pytest.mark.skip(reason="setting up CI")
def test_gyro_posterior_grid_cool():

    bounds_error = 'limit'
    age_grid = np.linspace(0, 2700, 101)
    Teff = 3820
    Prot = 2

    def run(N):

        print(f"{datetime.now().isoformat()} begin N={N}")
        age_post = gyro_age_posterior(
            Prot, Teff, age_grid=age_grid, bounds_error=bounds_error,
            N_grid=N, verbose=False
        )
        print(f"{datetime.now().isoformat()} end N={N}")
        return age_post

    plt.close("all")
    plt.plot(age_grid, run(128), label='N=128', lw=0.5)
    plt.plot(age_grid, run(256), label='N=256', lw=3)
    plt.plot(age_grid, run(512), label='N=512', lw=0.5)
    plt.plot(age_grid, run(1024), label='N=1024', lw=0.5)
    #TODO FIXME: formalize criterion for assessing "convergence" here.
    plt.legend(loc='best', fontsize='x-small')
    plt.title('teff 3820, prot 2 days, 100 age points')
    plt.xlabel("age")
    plt.ylabel("probability")
    plt.savefig('../results/debug/grid_size_check_cool.png', dpi=300)

if __name__ == "__main__":
    test_gyro_posterior_grid_sunlike()
    test_gyro_posterior_grid_cool()
