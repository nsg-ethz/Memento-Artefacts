"""Replay memory implementation for puffer evaluation."""

from typing import List, Optional

import numpy as np
from experiment_helpers.data import Path

import memento

from .data import puffer_sample_sizes
from .models import (load_model, predict_losses, predict_probabilities,
                     ttp_bin_centers, ttp_discretize)


class PufferMemory(memento.MultiMemory):
    """Combination of PufferReplayMemory and optional random ReservoirMemory.

    If `size` is provided, it will be the combined size of both memories,
    with the reservoir taking `reservoir_fraction` of it.
    """

    def __init__(self, *args,
                 size: Optional[int] = None,
                 random_fraction: float = 0.0,
                 random_forget: float = 0.0,
                 **puffer_mem_kwargs):
        assert random_fraction >= 0.0
        assert random_fraction <= 1.0
        memories: List[memento.bases.MemoryBase] = []
        weights = []

        if random_fraction < 1:
            puffer_fraction = 1 - random_fraction
            puffersize = (int(size * puffer_fraction)
                          if size is not None else None)
            memories.append(PufferMemento(
                *args, size=puffersize, **puffer_mem_kwargs
            ))
            weights.append(puffer_fraction)

        if random_fraction > 0:
            reservoirsize = (int(size * random_fraction)
                             if size is not None else None)
            memories.append(memento.alternatives.ReservoirMemory(
                size=reservoirsize, forget=random_forget
            ))
            weights.append(random_fraction)

        super().__init__(memories=memories, weights=weights)


class PufferMemento(memento.Memento):
    """Replay memory for the puffer model.

    We use the CategoricalOuptut helper to deal with the predicted labels and
    provide the Jensen-Shannon Distance, BBDRs as we have softmax predictions.

    For now, used model for next prediction as default (index 0).

    The predictor can be provided as a dir, in which case the model with
    index 0 is loaded from this path.
    """

    def __init__(self, *args, index: int = 0,
                 distances=None, sort=None,
                 workers=None,
                 binned_sort=False,
                 **kwargs):
        self.workers = workers
        self.default_modelfile = f"py-{index}.pt"

        if distances is None:
            distances = "both"
        if distances in ("both", "input", "output"):
            _distances = []
        else:
            _distances = distances

        if not _distances:
            _sort = ["x"]
            if distances in ("both", "input"):
                _distances.append(memento.JSDCategoricalDistribution(
                    'yhat', workers=self.workers))
                _sort = ["yhat", *_sort]
            if distances in ("both", "output"):
                _distances.append(memento.JSDCategoricalPoint(
                    'ydiscrete', classes=len(ttp_bin_centers()),
                    workers=self.workers))
                _sort = ["ydiscrete" if binned_sort else "y", *_sort]
            if sort is None:
                sort = _sort

        super().__init__(*args, distances=_distances,
                         batching_sort=sort, **kwargs)

    def insert_datadict(self, datadict):
        """Add discretized distances before insert."""
        datadict['ydiscrete'] = ttp_discretize(datadict['y'])
        return super().insert_datadict(datadict)

    def update_predictor(self, predictor):
        """Update the internal predictor.

        Accept model directory or path as well.
        """
        try:
            predictor = Path(predictor)  # Works for str and PathLike.
            if predictor.is_dir():
                predictor = predictor / self.default_modelfile
            predictor = load_model(predictor)
        except ValueError:
            pass  # Predictor is not a path.
        super().update_predictor(predictor)

    def predict(self, x):
        """Softmax prediction of workload."""
        return predict_probabilities(self.predictor, x, workers=self.workers)


class PufferRandomMemory(memento.utils.Random,
                         memento.alternatives.NoPredictorMemoryBase):
    """A random memory similar to the strategy used by the Puffer authors.

    Concretely, we have a fixed horizon (forget all data older than X days) adn
    within this window, we keep Y% of the data of each previous day.
    The Puffer authors use 14 days and 90% of the data.
    """

    def __init__(self, *args, horizon=14, fraction=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self.fraction = fraction

    def insert_datadict(self, datadict):
        """Add a counter to identify which day the data belongs to."""
        if not self.data:
            next_index = 0
        else:
            next_index = np.max(self.data['index']) + 1

        num_samples = memento.utils.sample_count(datadict)
        datadict['index'] = next_index * np.ones(num_samples)

        return super().insert_datadict(datadict)

    def select(self, max_samples, current_data, new_data):
        # Split data by index
        merged = memento.utils.merge_datadicts([current_data, new_data])
        # By definition, we will not have any gaps, so we can use a range to
        # get the last `horizon` indices.
        indices = np.arange(
            np.min(merged['index']), np.max(merged['index']) + 1
        )[-self.horizon:][::-1]  # Select final horizon and reverse.

        # Compute number of samples per timestep.
        all_samples = puffer_sample_sizes(total_samples=max_samples,
                                          discount=self.fraction,
                                          days=len(indices))

        # Selecte data.
        selected = []
        for index, samples in zip(indices, all_samples):
            index_samples = merged['index'] == index
            available = np.sum(index_samples)
            assert available >= samples
            selection = self.rng.choice(available, size=samples, replace=False)
            selected.append({
                key: value[index_samples][selection]
                for key, value in merged.items()
            })

        return memento.utils.merge_datadicts(selected)


class PufferLARS(memento.alternatives.LossAwareBalancedReservoir):
    """Puffer reduces the prediciton problem to classification, so try LARS.

    The predicted transmit time bin is the class.
    """

    def __init__(self, *args, index: int = 0, workers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.workers = workers
        self.default_modelfile = f"py-{index}.pt"

    def update_predictor(self, predictor):
        """Update the internal predictor.

        Accept model directory or path as well.
        """
        try:
            predictor = Path(predictor)  # Works for str and PathLike.
            if predictor.is_dir():
                predictor = predictor / self.default_modelfile
            predictor = load_model(predictor)
        except ValueError:
            pass  # Predictor is not a path.
        super().update_predictor(predictor)

    def predict(self, x):
        """Softmax prediction of bin probabilities."""
        return predict_probabilities(self.predictor, x, workers=self.workers)

    def classes(self, data):
        """Classes are discretized values."""
        return ttp_discretize(data['y'], model=self.predictor)

    def losses(self, data):
        """Compute loss per item."""
        if (self.predictor is None) and not self.require_predictor:
            return super().losses(data)
        return predict_losses(
            self.predictor, data['x'], data['y'], workers=self.workers)


class PufferSampleMetricBase(memento.alternatives.SampleMetricMemory):
    def __init__(self, *args,
                 index: int = 0, workers=None,
                 batching_sort=('y', 'yhat', 'x'), **kwargs):
        super().__init__(*args, batching_sort=list(batching_sort), **kwargs)
        self.workers = workers
        self.default_modelfile = f"py-{index}.pt"

    def update_predictor(self, predictor):
        """Update the internal predictor.

        Accept model directory or path as well.
        """
        try:
            predictor = Path(predictor)  # Works for str and PathLike.
            if predictor.is_dir():
                predictor = predictor / self.default_modelfile
            predictor = load_model(predictor)
        except ValueError:
            pass  # Predictor is not a path.
        super().update_predictor(predictor)

    def predict(self, x):
        """Softmax prediction of bin probabilities."""
        return predict_probabilities(self.predictor, x, workers=self.workers)


class PufferConfidence(PufferSampleMetricBase):
    """Confidence-based memory"""

    def metric(self, data):
        if (self.predictor is None) and not self.require_predictor:
            return super().metric(data)
        return predict_probabilities(
            self.predictor, data['x'], workers=self.workers).max(axis=1)


class PufferLoss(PufferSampleMetricBase):
    """Loss-based memory"""

    def metric(self, data):
        if (self.predictor is None) and not self.require_predictor:
            return super().metric(data)
        # Invert losses -> higher loss means to keep sample.
        return -1 * predict_losses(
            self.predictor, data['x'], data['y'], workers=self.workers)


class PufferClassCounts(PufferSampleMetricBase):
    """A simplified density based on class (aka label/bin) counts only."""

    def metric(self, data):
        bins = ttp_discretize(data['y'], model=self.predictor)
        total = len(bins)
        counts_per_bin = dict(zip(*np.unique(bins, return_counts=True)))

        # Class fraction = density, higher means easier forgotten.
        # We divide by the total count because otherwise softmax returns
        # only zeroes because the denominator is too large.
        return np.array(list(counts_per_bin[_bin] for _bin in bins)) / total


class PufferStalled(PufferSampleMetricBase):
    """Select samples that belong to stalled sessions.

    Requires the 'stalled' key in the data, e.g. set `load_stalls=True` in
    the PufferDataReplay.
    """

    def metric(self, data):
        """Low metric (keep) if stalled, high metric (forget) otherwise.

        We return 1 and 0; temperature scaling will take care of the rest.
        """
        # Read as: Return 0 if in stalled session else 1.
        return np.where(data['stalled'], 0., 1.)
