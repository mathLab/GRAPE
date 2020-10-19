Welcome to GRAPE's documentation!
===================================================

GRAph Parallel Environment

Description
--------------------
GRAPE (GRAph Parallel Environment) is a Python package that takes advantage of
Graph Theory into a High Performance Computing (HPC) environment to develope a
screening tool aimed at studying the effect of different kinds of perturbations
in interconnected systems, such as indsutrial plants.

The tool allows to represent the dependencies between components and predict
the state of health and the residual functionality of degradable systems after
a casualty, suggesting the proper reconfiguration strategies to mitigate the
damage. The results obtained from the graph analysis can be therefore used to
improve topology, robustness, and resilience profile of industrial facilities
against domino effect propagation.

In particular, the components contribution to the cascade effects resulting
from adverse events can be evaluated through centrality and efficiency
measures, highlighting the plants major criticalities, vulnerabilities and
potential weak points.

Considering that the most computationally expensive parts of the program
involve the calculation of shortest paths, parallelization of shortest path
computation in large unweighted graphs was integrated in the program.  This was
done taking advantage of the Python modules multiprocessing and threading.  Two
different sequential algorithms for the solution of the shortest path problem
have been parallelized including a Single Source Shortest Path (SSSP) algorythm
for sparse graphs and an All Pairs Shortest Path one (APSP) for dense graphs.


Installation
--------------------
The official distribution is on GitHub, and you can clone the repository using
::

    git clone https://github.com/mathLab/GRAPE

To install the package just type:
::

    python setup.py install

To uninstall the package you have to rerun the installation and record the installed files in order to remove them:

::

    python setup.py install --record installed_files.txt
    cat installed_files.txt | xargs rm -rf


Developer's Guide
--------------------

.. toctree::
   :maxdepth: 1

   code
   contact
   contributing
   LICENSE


Indices and tables
--------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

