<p align="center"><img src="docs/source/gyrointerp_logo.png" alt="gyrointerp" width="70%"/></p>

<img src="https://github.com/lgbouma/gyro-interp/workflows/Tests/badge.svg">

## Install
Preferred installation method:
```shell
pip install gyrointerp
```

Or 
```shell
git clone https://github.com/lgbouma/gyro-interp
cd gyro-interp
python setup.py install
```

External dependencies required to use the core functionality: `numpy`,
`pandas`, `astropy`, `astrobase`.

Additional dependencies required for (non-core) functionality: `astroquery`,
`[cdips](https://github.com/lgbouma/cdips)`.

## Verify installation
To verify your installation is working:
```shell
cd gyro-interp
pytest
```
which will execute the tests.  This takes about two minutes on my laptop.

## Example
Given a star's rotation period, effective temperature, and uncertainties, what
is the gyrochronological age posterior over a grid spanning 0 to 2.6 Gyr?

```
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
```


## Caveats

### Stellar Evolution
This code models the evolution of rotation periods for main-sequence stars with
temperatures of 3800-6200 K (masses of 0.5-1.2 solar).  This model is
calibrated based on cluster data from 0.08-2.6 Gyr.  At younger ages, rotation
periods are less predictive of age, and other age indicators may be more
useful.  At older ages, the effects of stellar evolution begin to become
important, especially for the more massive stars.

If you have auxiliary data, such as stellar surface gravities derived from
spectra, they can help determine whether this model is applicable to your star.

### Binarity
Before applying gyrochronology, it's worth asking whether your star is
potentially a binary.  If it is, you should be concerned about that affecting
the rotation period, both due to potential biases in temperature and rotation
period measurements, and due to the physical effects that even a
widely-separated companion can have on a star's spin-down.

A few ways to determine whether your star might be a binary include:

* Obtain high resolution images, and determine whether there are nearby
companions.  You can also check the Gaia point source catalog for such
companions, though you may not achieve the same contrast limits close to the
star.

* Check the Gaia point source catalog for whether the renormalized unit weight
error (RUWE) exceeds some threshold.  Additional astrometric scatter, which is
what this quantity measures, can be caused by either a marginally resolved
point source, or by astrometric binarity.

* Obtain spectra, and check if they are double-lined, or if they show
significant radial variations over time.  The `radial_velocity_error` column
in Gaia DR3 can help diagnose the latter case, though it also scales with 

* Query the Gaia point source catalog in a local spatial volume around your
star.  With the resulting sample of stars, check whether your star is an
outlier in the HR diagram.  This can be an indication of photometric binarity.

Generally speaking, the best approaches will differ based on your stars of
interest.
A few separate utilities that can help in assessing these types of utilties are
available through
`[astroquery.gaia](https://astroquery.readthedocs.io/en/latest/gaia/gaia.html)`, 
and `[cdips.utils.gaiaqueries](https://github.com/lgbouma/cdips)`.
