# Memento  <!-- omit in toc -->

This repository allows to run all experiments and create all plots for the paper _On Sample Selection for Continual Learning: a Video Streaming Case Study_, where we introduce **Memento**, a system to manage a training sample set over time and decide when retraining models is necessary.
Memento is based on the idea of maxmimizing the coverage of sample space, retraining rare samples and improving tail performance.
It achieves this by estimating the sample-space density.

Running all experiments requires a very long time and large amount of storage.
Thus, we have prepared a small demonstration in addition to the main experiments. The demo uses only a week of data, but the same code otherwise.

> To evaluate the retraining events and model performance for the Memento deployment on Puffer, the actual models and retraining statistics are required. To make this possible during artifact evaluation, we provide this data as part of this repo under `results/memento-deployment`.


## Table of Contents  <!-- omit in toc -->

- [Installation](#installation)
- [Demo](#demo)
- [Experiments](#experiments)
- [Code overview](#code-overview)


## Installation

The latet code was tested Python 3.10, but you should be able to run it with Python 3.9 and above. We recommend using a virtual environment.

```bash
# Create and activate virtual environment.
python -m venv env
source env/bin/activate
# Install dependencies.
pip install -r requirements.txt
```

## Demo

Running all experiments takes several hundred GB to TB of disk space and requires several days to complete, as we need to evaluate data from many days and need to process them sequentially to evaluate to effects of continual learning over time.
While the individual experiments can be parallelized, they still take hours each.

To get a quicker understanding of how the experiments work and what results we collect, we have prepared a short demonstration running with only a week of data.
You can run the demo wtih the following commands.

```bash
# Download and parse data.
./demo.py download
# Analyze ABR algorithms, should take 1-2 hours.
./demo.py run-demo analysis
# Replay data, testing continual learning. Trains models and takes several
# hours with a GPU, and longer without.
./demo.py run-demo replay
```

After running either the analysis or replay, you can use the [demo notebook](notebooks/demo.ipynb) to check out and visualize the results.

You can compare the code in [demo.py](demo.py) to the Puffer experiments in [experiments/puffer/experiments.py](experiments/puffer/experiments.py).
While the main experiments use longer time ranges and test more parameters, you will see that they use the same experiment functions.


## Experiments

If you want to run all experiments, use the `./run.py` command and the
remaining notebooks in [notebooks/](notebooks/).
The individual notebooks contain an overview on which figures from the paper are created with them and which experiments you need to run to get the required data.

- [puffer-analysis](notebooks/puffer-analysis.ipynb): evaluation of the data published by the Puffer project, and all related figures.
- [puffer-replay](notebooks/puffer-replay.ipynb): evaluation of replaying the Puffer data offline to test Memento's parameters.
- [puffer-extra](notebooks/puffer-extra.ipynb): additional investigation of density-based sample selection.
- [ns3-replay](notebooks/ns3-replay.ipynb): replay simulation data to evaluate Memento's behavior under artificial traffic shifts.

The experiments are organized in groups, which you can see by running:

```bash
./run.py --help                         # Show all groups.
./run.py puffer-experiments --help      # Show help per group.
```

All constants, paths, etc. are defined in config files.
If you want to override any configuration values, simply add them to the [user config in config.py](config.py).
That is, you can redefine any value you find in one of the experiment config files here, and your user config will take precedence.


### Useful parameters to `./run.py`  <!-- omit in toc -->

If you do not want to run all experiments at once, you can use UNIX-style glob
pattern matching. First, you can run the command with the `-l` flag to print
all available experiments, e.g.:

```bash
./run.py puffer-analysis -l
```

In the case of `puffer-analysis`, experiments are organized by day in the format `YYYY-MM-DD`.
You can match experiments , e.g. selecting only July 2022:


```bash
./run.py puffer-analysis "2022-07-*"
```

In general, you can control the verbosity of experiments using `-v` flags, e.g.:

```bash
./run.py puffer-analysis        # Only warnings and errors.
./run.py puffer-analysis -v     # General info in addition.
./run.py puffer-analysis -vv    # Full debugging output.
```

You can also run multiple experiments in parallel using the `-j` option and
control how many cores each experiment uses with the `-w` option.
For example, to run five experiments in parallel using up to 10 cores each, run:

```bash
./run.py puffer-analysis -j 5 -w 10
```

Finally, Slurm is supported if you require it to schedule GPU jobs by providing
the `--slurm` flag. You can use the `-j` flag to set a maximum number of
concurrent jobs and again use `-w` to control the cores per job, e.g.:

```bash
./run.py puffer-analysis --slurm -w 10       # Run on slurm with 10 workers each
./run.py puffer-analysis --slurm -j 5 -w 10  # At most 5 concurrent jobs.
```

You can set constraints and resource usage in [config.py](config.py)


## Code overview

The experiments for Puffer and ns-3 data can be found in [experiments/](experiments/).
Both have the same layout, containing a `config.py` file for constants, and `experiments.py` file where the actual experiments are listed, and and `implementation` directory with the experiment code.

As described above, the [notebooks/](notebooks) directory contains all Jupyer notebooks to collect and visualize the results.

The [memento/](memento/) directory contains the code and tests for Memento, which is implemented as a sample memory.
It includes some alternative memories (e.g. using random selection).

The [ns3-simulation/](ns3-simulation/) directory contains the ns-3 code to run the simulations to create the ns-3 data.
We have already pre-run the simulations and included the data.
Follow the instructions in [ns3-simulation/README.md](ns3-simulation/README.md) to recreate them.

Finally, the [python-experiment-helpers/](python-experiment-helpers/) directory contains the code that provides the CLI running the experiments and additional utilies for data reading and writing, logging, and plotting.
