Examples
========================================

Gyrochronal age for one star
++++++++++++++++++++

Given a single star's rotation period, effective temperature, and
uncertainties, what is the gyrochronological age posterior over a grid spanning
0 to 2.6 Gyr?

.. code-block:: python

  import numpy as np
  from gyrointerp import gyro_age_posterior

  Prot, Prot_err = 6, 0.2
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

This takes about 30 seconds to run on my laptop.  You can then plot the
posterior using matplotlib:

.. code-block:: python

  import matplotlib.pyplot as plt

  fig, ax = plt.subplots()
  ax.plot(age_grid, 1e3*age_posterior, c='k', lw=1)
  ax.update({
      'xlabel': 'Age [Myr]',
      'ylabel': 'Probability ($10^{-3}\,$Myr$^{-1}$)',
      'xlim': [0, 1000]
  })
  plt.show()


Gyrochronal ages for many stars
++++++++++++++++++++

Given the rotation periods, temperatures, and uncertainties for many stars,
what are the implied age posteriors?

.. code-block:: python

  import os
  import numpy as np, pandas as pd
  from gyrointerp import gyro_age_posterior_list, get_summary_statistics

  def main():

      N_stars = os.cpu_count()
      Teffs = np.linspace(4000, 5500, N_stars)
      Teff_errs = 100 *np.ones(N_stars)
      # at >~20 days, assume a few percent relative uncertainty on periods
      Prots = np.linspace(15, 22, N_stars)
      Prot_errs = 0.03 * Prots

      # The output posteriors will be cached at ~/.gyrointerp_cache/{cache_id}
      cache_id = 'my_awesome_stars'

      # That 5500 K star with Prot = 22 days is near the Ruprecht-147 sequence.
      # Let's extend the age_grid up to 4000 Myr (4 Gyr); the extrapolation 
      # past 2.6 Gyr is based on the M67 data.
      age_grid = np.linspace(0, 4000, 500)

      # Let's pass optional star IDs to name the posterior csv files.
      star_ids = [f"FOO{ix}" for ix in range(N_stars)]

      csvpaths = gyro_age_posterior_list(
          cache_id, Prots, Teffs, Prot_errs=Prot_errs, Teff_errs=Teff_errs,
          star_ids=star_ids, age_grid=age_grid, bounds_error="4gyrlimit",
          interp_method="pchip_m67"
      )

      # Read the posteriors and print their summary statistics.
      for csvpath, Prot, Teff in zip(sorted(csvpaths), Prots, Teffs):
          df = pd.read_csv(csvpath)
          r = get_summary_statistics(df.age_grid, df.age_post)
          msg = f"Age = {r['median']} +{r['+1sigma']} -{r['-1sigma']} Myr."
          print(f"Teff {int(Teff)} Prot {Prot:.2f} {msg}")

  if __name__ == "__main__":
      main()

In this example we guarded the multiprocessing being executed in
*gyro_age_posterior_list* in a *__main__* block, per the suggestion in the
`multiprocessing docs
<https://docs.python.org/3/library/multiprocessing.html>`_.  This example also
takes about 30 seconds to run on my laptop, so the multithreading is doing what
we want.


Auxiliary tools
++++++++++++++++++++

**Comparing a single star's rotation period against open cluster populations**
(TODO: add this plot)

**Plotting the gyro posterior**
(TODO: embed plot)


