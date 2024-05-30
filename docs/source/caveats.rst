Caveats
========================================

Stellar Evolution
++++++++++++++++++++

This code models the ensemble evolution of rotation periods for main-sequence
stars with temperatures of 3800-6200 K (masses of 0.5-1.2 solar).  The 
calibration data for the model span 0.08-4 Gyr.  At younger ages, rotation
periods are less predictive of age, and other age indicators may be more
useful.  At older ages, the effects of stellar evolution become important,
especially for the more massive stars.

If you have auxiliary data, such as stellar surface gravities derived from
spectra, they can help determine whether your star might have undergone
significant nuclear evolution -- in other words whether it is a subgiant or
even a giant.  If it is, this model is not applicable.


Binarity
++++++++++++++++++++

Before applying gyrochronology, it's worth asking whether your star might be a
binary.  If it is, you should proceed carefully.  Binarity can observationally
bias temperature and rotation period measurements.  There is also a physical
concern that even a widely-separated companion might influence a star's
spin-down, by encouraging early dispersal of the protostellar disk.

A few ways to determine whether your star might be a binary include:

* Get high resolution images, and determine whether there are nearby
  companions.  You can also check the Gaia point source catalog for such
  companions, though you may not achieve the same contrast limits close to the
  star.

* Check the Gaia point source catalog for whether the renormalized unit weight
  error (RUWE) exceeds some threshold.  Additional astrometric scatter, which is
  what this quantity measures, can be caused by either a marginally resolved
  point source, or by astrometric binarity.

* Obtain spectra, and check if they are double-lined, or if they show
  significant radial variations over time.  The *radial_velocity_error* column
  in Gaia DR3 can help diagnose the latter case, although care should be taken
  for stars that are faint and have low S/N spectra.

* Query the Gaia point source catalog in a local spatial volume around your
  star.  With the resulting sample of stars, check whether your star is an
  outlier in the HR diagram.  This can be an indication of photometric binarity.

As mentioned in the section on :ref:`visual interpolation`, the same types of
considerations apply to hot Jupiter systems, or any kinds of systems in which
tidal effects might alter the star's rotation period.

Generally speaking, the best approaches will differ based on your stars of
interest.  A few separate utilities that can help in assessing these types of
utilties are available through
`astroquery.gaia <https://astroquery.readthedocs.io/en/latest/gaia/gaia.html>`_,
and `cdips.utils.gaiaqueries <https://github.com/lgbouma/cdips>`_.
Both are pip-installable.
