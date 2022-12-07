import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gyrointerp.gyro_posterior import gyro_age_posterior
from datetime import datetime

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

plt.plot(age_grid, run(64), label='N=64', lw=0.5)
plt.plot(age_grid, run(128), label='N=128', lw=0.5)
plt.plot(age_grid, run(256), label='N=256', lw=3)
plt.plot(age_grid, run(512), label='N=512', lw=0.5)
plt.legend(loc='best', fontsize='x-small')
plt.title('teff 5500, prot 10 days, 100 age points')
plt.xlabel("age")
plt.ylabel("probability")
plt.savefig('../results/age_posteriors/grid_size_check.png', dpi=300)
