from setuptools import setup, find_packages
import grape

# Package meta-data.
NAME = grape.__title__
DESCRIPTION = 'GRAph Parallel Environment.'
URL = 'https://github.com/mathLab/GRAPE'
MAIL = grape.__mail__
AUTHOR = grape.__author__
VERSION = grape.__version__
KEYWORDS = 'risk-analysis graph-theory parallel-programming shortest-path system-analysis fault-diagnostics'

REQUIRED = [
    'networkx', 'numpy', 'scipy', 'matplotlib',
]

EXTRAS = {
    'docs': ['sphinx', 'sphinx_rtd_theme'],
}

LDESCRIPTION = (
    "GRAPE is a Python package that takes advantage of Graph Theory "
    "into a High Performance Computing (HPC) environment to develop a screening tool aimed "
    "at studying the effect of different kinds of perturbations in interconnected systems, such as"
    "indsutrial plants."
    "\n"
    "The tool allows to represent the dependencies between components and predict the state"
    "of health and the residual functionality of degradable systems after a damage"
    ", suggesting the proper reconfiguration strategies to mitigate it."
    "The results obtained from the graph analysis can be therefore used to improve topology,"
    " robustness, and resilience profile of industrial facilities against domino effect propagation."

    "In particular, the components contribution to the cascade effects resulting from adverse"
    "events can be evaluated through centrality and efficiency measures, highlighting the "
    "plants major criticalities, vulnerabilities and potential weak points.\n"

    "Considering that the most computationally expensive parts of the program involve the"
    "calculation of shortest paths, parallelization of shortest path computation in large "
    "unweighted graphs was integrated in the program."
    "This was done taking advantage of the Python module multiprocessing."
    "Two different sequential algorithms for the solution of the shortest path problem have been"
    "parallelized including a Single Source Shortest Path (SSSP) algorythm for sparse graphs"
    "and an All Pairs Shortest Path one (APSP) for dense graphs."
    "\n"
)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LDESCRIPTION,
    author=AUTHOR,
    author_email=MAIL,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords=KEYWORDS,
    url=URL,
    license='MIT',
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    zip_safe=False
)
