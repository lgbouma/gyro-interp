Frequently asked questions
========================================

What happens for stars cooler than 3800 K?
+++++++++++++++++++++++++++++++++++++++++++

For stars with measured effective temperatures outside of our calibrated range
(3800-6200 K), we have opted to force our model to return ``NaN`` age
posteriors.  For stars near the boundary (e.g., 3800+/-100 K) for which a
portion of the likelihood is outside our nominal temperature range, the
returned posterior includes that region through a local extrapolation of the
mean polynomial models discussed in BPH23.

The same point applies for stars hotter than 6200 K.

What happens for stars younger than 80 Myr?
+++++++++++++++++++++++++++++++++++++++++++

For stars younger than 0.08 Gyr, we fixed our "mean rotation period model" to
equal the lowest reference polynomial rotation period values, as set by the α
Per cluster.  This yields posterior distributions that are uniformly
distributed at ages younger than 0.08 Gyr.  So, the age posteriors for such
systems will be upper limits, which may or may not be useful for you!

Once stars begin getting this young, there are other age-dating techniques that
may be more constraining.  For instance, you might search spectra for lithium
absorption, check broadband photometry for near-infrared excesses, or
analyze HR diagrams for evidence of pre-main-sequence evolution.

What happens for stars older than 2.6 Gyr?
+++++++++++++++++++++++++++++++++++++++++++

For stars older than 2.6 Gyr, we have implemented a few possible extrapolation
approaches.  Our default adopted approach is to say "please do not try to
compute ages for such stars", because their spindown rates are currently less
well-constrained by the data.  This default decision corresponds to calling
``gyro_age_posteriors`` with ``interp_method=='pchip_m67'`` and
``bounds_error=='4gyrlimit'``.
This yields posterior probability distributions
that are uniformly distributed at ages older than 4 Gyr - in other words, these
uncertainties are biased near the 4 Gyr boundary!

If you really wish to push the old age limit, this can be done up to 4 Gyr by
setting ``interp_method=='pchip_m67'`` and ``bounds_error=='4gyrextrap'``.  For
the invested user, the origin of this setting is discussed in `this
documentation note
<https://docs.google.com/document/d/1X_tOf1y1e8yvRZFo7NgPTsOSSR5p2J1wsyb1NT3DDB4/edit?usp=sharing>`_.
This gives accurate and unbiased ages that reproduce the cluster data up to 4
Gyr.  Beyond 4 Gyr, other age-dating methods might be a safer choice.


Are there movies?
+++++++++++++++++++++++++++++++++++++++++++

Yes!  `This movie <https://lgbouma.com/movies/model_prot_vs_teff.mp4>`_ shows
random draws from the model over the first two gigayears.  `This movie
<https://lgbouma.com/movies/prot_teff_model_data.gif>`_ compares these random
draws to available cluster data at fixed timesteps.
