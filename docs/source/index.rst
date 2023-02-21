gyro-interp
++++++++++++

Hi!  Welcome to the documentation for ``gyrointerp``, a Python package that
calculates gyrochronal ages by interpolating between open cluster rotation
sequences.

``gyrointerp`` packages the model from Bouma, Palumbo & Hillenbrand (2023) 
into a fast and easy-to-use framework.  The documentation below will walk you
through the most common use-cases.  For brevity, we'll refer to that paper as
BPH23.

This package is designed to meet the needs of working astronomers with
interests in gyrochronal age measurement, and we encourage community
involvement.  If you find a bug or would like to request a feature, please do
create an `issue on Github <https://github.com/lgbouma/gyro-interp>`_. 

.. |br| raw:: html

   <br />

.. image:: gyrointerp_logo.png
   :width: 50%
   :align: center

Attribution:
++++++++++++

The reference for both the software and method will be Bouma, Palumbo &
Hillenbrand (2023).  This manuscript is currently under review at the AAS
journals; the submitted version is available `at this
link <https://lgbouma.com/pdfs/BPH23_preprint.pdf>`_.

The references for the rotation period data upon which the code is based are
as follows.

* α Per: `Boyle & Bouma (2023) <https://ui.adsabs.harvard.edu/abs/2022arXiv221109822B/abstract>`_
* Pleiades: `Rebull et al. (2016) <https://ui.adsabs.harvard.edu/abs/2016AJ....152..113R/abstract>`_
* Blanco-1: `Gillen et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.1008G/abstract>`_
* Psc-Eri: `Curtis et al. (2019a) <https://ui.adsabs.harvard.edu/abs/2019AJ....158...77C/abstract>`_
* NGC-3532: `Fritzewski et al. (2022) <https://ui.adsabs.harvard.edu/abs/2021A%26A...652A..60F/abstract>`_
* Group-X: `Messina et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022A%26A...657L...3M/abstract>`_
* Praesepe: `Rampalli et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJ...921..167R/abstract>`_
* NGC-6811: `Curtis et al. (2019b) <https://ui.adsabs.harvard.edu/abs/2019ApJ...879...49C/abstract>`_
* NGC-6819: `Meibom et al. (2015) <https://ui.adsabs.harvard.edu/abs/2015Natur.517..589M/abstract>`_
* Ruprecht-147 `Curtis et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020ApJ...904..140C/abstract>`_
* M67: `Barnes et al. (2016) <https://ui.adsabs.harvard.edu/abs/2016ApJ...823...16B/abstract>`_ and `Dungee et al (2022) <https://ui.adsabs.harvard.edu/abs/2022ApJ...938..118D/abstract>`_.


User Guide:
++++++++++++

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    installation
    examples
    caveats
    faqs
    gyrointerp

Changelog:
++++++++++

**0.2 (2023-02-21)**
* Initial software release to PyPI and github.

**0.1 (2023-02-21)**
* Initial software release to github.
