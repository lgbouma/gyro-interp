Examples
============

Gyrochronal ages
++++++++++++

Given a star's rotation period, effective temperature, and uncertainties, what
is the gyrochronological age posterior over a grid spanning 0 to 2.6 Gyr?

.. code-block:: python

  import numpy as np
  from gyrointerp import gyro_age_posterior
  Prot, Prot_err = 6, 0.05
  Teff, Teff_err = 5500, 100
  age_grid = np.linspace(0, 2600, 100)

  # calculate the gyro-age posterior over the specified grid
  age_posterior = gyro_age_posterior(
    Prot, Teff, Prot_err=Prot_err, Teff_err=Teff_err, age_grid=age_grid
  )

  from gyrointerp.helpers import given_grid_post_get_summary_statistics

  # get a dictionary containing summary statistics from the age posterior
  result = given_grid_post_get_summary_statistics(age_grid, age_posterior)
  print(result)


Auxiliary tools
++++++++++++

**Comparing a single star's rotation period against open cluster populations**
(TODO: add this plot)

**Plotting the gyro posterior**
(TODO: embed plot)

**Checking for binarity**

Before applying gyrochronology, it's worth asking whether your star is
potentially a binary.  If is, you should be concerned about that affecting the
rotation period, due to a story involving disk dispersal that's been described
by eg., `Meibom et al. 2007 <https://ui.adsabs.harvard.edu/abs/2007ApJ...665L.155M/>`_,
and has been borne out by more data since then.

.. code-block:: python

  from gyroemp.binary_checker import (
      given_source_ids_return_possible_binarity
  )

  # given a Gaia DR2 or DR3 source_id, determine whether the star is potentially
  # binary

  source_id = '446488105559389568'
  gaia_datarelease = 'gaiadr3'
  target_df, nbhr_df = given_source_id_return_possible_binarity(
    source_id, gaia_datarelease
  )
