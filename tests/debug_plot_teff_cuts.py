from gyrointerp.models import teff_0, teff_zams, logistic
import numpy as np, matplotlib.pyplot as plt
import os

outdir = "../results/debug/"
if not os.path.exists(outdir): os.mkdir(outdir)

age = np.linspace(0, 1500, 1000)
plt.plot(
    age, teff_0(age), label='teff_0 by eye'
)
plt.plot(
    age, teff_zams(age), label='teff_zams MIST'
)
plt.hlines(3800, 0, 1500, ls='--', zorder=-1, lw=0.5, color='darkgray',
           label='gyro model cutoff')
plt.xlabel('Time [years]')
plt.ylabel(r'T$_{\rm eff}$ of logistic transition [K]')
plt.xscale('log')
plt.xlim((50,1200))
plt.legend()
outpath = os.path.join(outdir, 'teff_cuts_vs_time.png')
plt.savefig(outpath, dpi=400)

print(teff_zams(80))
print(teff_zams(120))
print(teff_zams(300))


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
outpath = os.path.join(outdir, 'logistic_teff0-4500.png')
plt.savefig(outpath, dpi=400)
