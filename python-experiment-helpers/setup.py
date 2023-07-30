"""Experiment helper setup."""

from distutils.core import setup

setup(
    name='experiment_helpers',
    description="Experiment helper scripts.",
    author="Alexander Dietmüller",
    author_email="adietmue@ethz.ch",
    url=("https://gitlab.ethz.ch/nsg/employees/adietmue/projects/"
         "python-experiment-helpers"),
    version='0.2.0',
    packages=["experiment_helpers"],
    install_requires=[
        'click',
        'click-pathlib',
        'click-log',
        'cloudpickle',
        'requests',
        'numpy',
        'seaborn',
        'watchdog',
        'universal-pathlib',
        'sshfs',
    ],
)
