# Memento

A replay memory maximizng the coverage of sample space by estimating the
sample-space density,

The memory needs to be initialized with a functions that compute the distance between batches of samples.
Multiple distances can be combined, e.g. a distance between predictions and a distance between (groud truth) outputs,
as used in the paper.

You can insert samples with the `.insert` method and get the current samples with the `.get` method.

For examples on how to instantiate the Memory, check out the the `memory.py` files in the implementation directories.
- [Memento for ns-3 models](../experiments/ns3/implementation/memory.py)
- [Memento for Puffer models](../experiments/puffer/implementation/memory.py)

For more details on all parameters, check the [bases.py](memento/memento/bases.py) and [models.py](memento/memento/models.py).
