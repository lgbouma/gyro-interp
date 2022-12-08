.. _installation:

Installation
============

``gyrointerp`` works in Python>3.6.  To use the code, first make sure you have
the latest versions of the "standard astro stack" installed. With ``pip``, you
can do this with the command:

.. code-block:: bash
  
    $ pip install numpy astropy pandas scipy --upgrade

Next, install ``gyrointerp``:

.. code-block:: bash
  
    $ pip install gyrointerp

We recommend installing and running ``gyrointerp`` in a ``conda`` virtual
environment. Install ``anaconda`` or ``miniconda`` `here
<https://conda.io/miniconda.html>`_, then see instructions `here
<https://conda.io/docs/user-guide/tasks/manage-environments.html>`_ to learn
more about ``conda`` virtual environments.

If you just want to know a star's gyrochronal age given its rotation period and
effective temperature, the installation above is sufficient.  However, the full
set of available functionality in the package includes pre-cooked queries to
Gaia DR3 to check whether a given star is likely to be a binary.  If you wish
to run such queries, you will also need to install the ``astroquery`` and
``cdips`` libraries:

.. code-block:: bash
  
    $ pip install astroquery cdips --upgrade


**Issues?**

If you run into any issues installing ``gyrointerp``, please create an `issue
on Github <https://github.com/lgbouma/gyro-interp>`_. 
