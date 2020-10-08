Installing pypbomb
==================

Owing to some problems I have had using conda-build, pypbomb is
installed using pip. However, cantera and pint are required, which
means you should have conda installed. Don't forget to activate the correct
conda environment before installing pypbomb!

Instructions
------------

#. Download and extracted the latest release from
   `github <https://github.com/cartemic/pypbomb/releases/latest>`_ and navigate
   to the top level directory of pypbomb (e.g. ``~/Downloads/pypbomb/``) in your
   terminal of choice.

#. Install cantera:

   .. code-block:: bash

      $ conda install -c cantera cantera

#. Install pint:

   .. code-block:: bash

      $ conda install -c conda-forge pint

#. Install pypbomb:

   .. code-block:: bash

      $ pip install .

#. You may now delete the pypbomb directory that you downloaded

Alternatively, if you want to tweak pypbomb while using it for your own ends,
move the pypbomb directory to where you want it, perform steps 1-3, and install
pypbomb in development mode:

.. code-block:: bash

   $ conda develop .

In this case, *do not* delete the downloaded pypbomb directory, since this is where pypbomb lives now.


