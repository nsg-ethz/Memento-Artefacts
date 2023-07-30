# Ns-3 experiments for Memento

- `config.py` contains the ns-3 specific configuration defaults.
- `experiments.py` containts the list of experiments that we run along with
  their parameters, see README how to run them.
- the `implementation` directory contains the model, memory and experiment
  code.

Note that `config.py` does not contain user-specific config like secrets or
storage dirs on the server. These are defined in the user-config in the root
of the directory and will be loaded at runtime. See main README for details.
