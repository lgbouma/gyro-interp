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
absorption, searching broadband photometry for near-infrared excesses, or
analyze HR diagrams for evidence of pre-main-sequence evolution.

What happens for stars older than 2.6 Gyr?
+++++++++++++++++++++++++++++++++++++++++++

For stars older than 2.6 Gyr, we have implemented a few possible extrapolation
approaches.  Our default adopted approach, the ``pchip_m67`` extrapolation,
provides one plausible interpolation between 2.6 and 4 Gyr based on the M67
data, though it is subject to larger systematic errors than e.g., our model
between 1 and 2.6 Gyr because the change of the slope in rotation period versus
time is not as well-constrained.  After 4 Gyr, as for α Per we simply force the
mean model’s rotation period to equal the highest reference rotation period
values, but now as set by M67.  This yields posterior probability distributions
that are uniformly distributed at ages older than 4 Gyr.

So, the age posteriors for such systems will be lower limits, and they may or
may not be useful for you!
