from agetools.models import teff_0, logistic
import numpy as np, matplotlib.pyplot as plt

age = np.linspace(115, 1000, 1000)
plt.plot(
    age, teff_0(age)
)
plt.savefig('../results/debug/teff0_vs_time.png')

teff_grid = np.linspace(3800, 6200, 1000)
plt.close("all")
plt.plot(
    teff_grid, 1-logistic(teff_grid, 4500, L=1, k=0.1), label='k=0.1'
)
plt.plot(
    teff_grid, 1-logistic(teff_grid, 4500, L=1, k=np.e**-3), label='e^-3'
)
plt.plot(
    teff_grid, 1-logistic(teff_grid, 4500, L=1, k=np.e**-4), label='e^-4'
)
plt.plot(
    teff_grid, 1-logistic(teff_grid, 4500, L=1, k=np.e**-5), label='e^-5'
)
plt.legend()
plt.savefig('../results/debug/logistic_teff0-4500.png')
