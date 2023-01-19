Examples
========================================

Gyrochronal ages
++++++++++++++++++++

Given a star's rotation period, effective temperature, and uncertainties, what
is the gyrochronological age posterior over a grid spanning 0 to 2.6 Gyr?

.. code-block:: python

  import numpy as np
  from gyrointerp import gyro_age_posterior

  Prot, Prot_err = 6, 0.1
  Teff, Teff_err = 5500, 100
  age_grid = np.linspace(0, 2600, 500)

  # calculate the age posterior at each age in `age_grid`
  age_posterior = gyro_age_posterior(
    Prot, Teff, Prot_err=Prot_err, Teff_err=Teff_err, age_grid=age_grid
  )

  # calculate dictionary of summary statistics
  from gyrointerp.helpers import get_summary_statistics
  result = get_summary_statistics(age_grid, age_posterior)

  print(f"Age = {result['median']} +{result['+1sigma']} -{result['-1sigma']} Myr.")


Auxiliary tools
++++++++++++++++++++

**Comparing a single star's rotation period against open cluster populations**
(TODO: add this plot)

**Plotting the gyro posterior**
(TODO: embed plot)
