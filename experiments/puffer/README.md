# Puffer experiments for Memento

- `config.py` defines the Puffer configuration defaults. See main README on how
  to provide a user config.
- `experiments.py` defines all Puffer experiments, see README how to run them.
- `implementation` contains all the actual experiment code.

Note that `config.py` does not contain user-specific config like secrets or
storage dirs on the server. These are defined in the user-config in the root
of the directory and will be loaded at runtime. See main README for details.
