# Ns-3 simulations for specified traffic patterns.

Distribution shift experiments with ns-3.

## Overview

- `generator/` An ns-3 module containing simulation helpers to generate traffic with specific CDF workloads.
- `simulation/` Simulation scripts.

## Running the simulations

### Docker with Visual Studio Code

When opening the directory, VSCode will prompt you to open the folder in a
container. Inside the dev container, you can run scripts from the local directory or use the `waf` command.
Before you run the simulations for the first time, you need to configure ns-3:

    $ waf configure

Afterwards, you can run everything:

    $ run_experiments.sh

Or individual experiments (you can append parameters):

    $ waf --run trafficgen
    $ waf --run "trafficgen --w1=0"

Check all available parameters:

    $ waf --run "trafficgen --PrintHelp"

### Docker

Use the `./docker-run.sh` script to run the simulations in a ns-3 environment.
The first time you run the script, it will take some time to download and start
the container. Afterwards, the container will be re-used.
The project directory will be mapped to the container.
Before you run the simulations for the first time, you need to configure ns-3:

    $ ./docker-run.sh waf configure

Afterwards, you can run everything:

    $ ./docker-run.sh run_experiments.sh

Or individual experiments (you can append parameters):

    $ ./docker-run.sh waf --run trafficgen
    $ ./docker-run.sh waf --run "trafficgen --w1=0"

Check all available parameters:

    $ ./docker-run.sh waf --run "trafficgen --PrintHelp"

If you need to clean up and remove the container, use `docker rm --force fitnets-runner`.
