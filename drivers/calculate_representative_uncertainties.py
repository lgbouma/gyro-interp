"""
Calculate the uncertainties for repesentative Teffs and ages quoted in the
abstract, and throughout the manuscript.  Writes to stdout.
"""

import numpy as np
from gyrointerp import gyro_age_posterior
from gyrointerp.models import slow_sequence
from gyrointerp.helpers import get_summary_statistics

# Sun-like stars
age_grid = np.linspace(0, 2600, 100)
ages = [200, 500, 1500, 2000]
Teffs, Teff_err = [5800, 5000, 4200, 4000, 3800], 50
for Teff in Teffs:
    print(42*'-')
    for age in ages:
        Prot = slow_sequence(Teff, age)
        Prot_err = 0.01*Prot
        age_posterior = gyro_age_posterior(
          Prot, Teff, Prot_err=Prot_err, Teff_err=Teff_err, age_grid=age_grid
        )
        d = get_summary_statistics(age_grid, age_posterior)
        print(f"Teff: {Teff}, age: {age} Myr, med: {d['median']:.1f}, "+
              f"+1sig%: {100*d['+1sigmapct']:.1f}%, -1sig%: {100*d['-1sigmapct']:.1f}%")
