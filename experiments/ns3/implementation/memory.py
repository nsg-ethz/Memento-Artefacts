"""Replay memory implementation for ns-3 simulation."""

import abc
from typing import List, Optional, Type

import memento

from ..config import Ns3ExperimentConfig


class _Ns3MultiMemory(memento.MultiMemory, abc.ABC):
    """Combination of Memento and an optional ReservoirMemory.

    If `size` is provided, it will be the combined size of both memories,
    with the reservoir taking `reservoir_fraction` of it.
    """
    @property
    @abc.abstractmethod
    def coverage_mem_class(self) -> Type[memento.Memento]:
        """The coverage memory class"""

    def __init__(self, *args,
                 size: Optional[int] = None,
                 random_fraction: float = 0.0,
                 random_forget: float = 0.0,
                 **mem_kwargs):
        assert random_fraction >= 0.0
        assert random_fraction <= 1.0
        memories: List[memento.bases.MemoryBase] = []
        weights = []

        if random_fraction < 1:
            cov_fraction = 1 - random_fraction
            cov_size = (int(size * cov_fraction)
                        if size is not None else None)
            memories.append(self.coverage_mem_class(
                *args, size=cov_size, **mem_kwargs
            ))
            weights.append(cov_fraction)

        if random_fraction > 0:
            reservoirsize = (int(size * random_fraction)
                             if size is not None else None)
            memories.append(memento.alternatives.ReservoirMemory(
                size=reservoirsize, forget=random_forget
            ))
            weights.append(random_fraction)

        super().__init__(memories=memories, weights=weights)


class WorkloadMemoryCoverage(memento.Memento):
    """Replay memory for the workload model."""

    def __init__(self, *args,
                 distances=None, sort=None, config=None, **kwargs):
        config = Ns3ExperimentConfig.with_updates(config)
        if distances in ("input", "output", "both", None):
            n_classes = config.workloads
            _distances = []
            _sort = []
            if distances in ("output", "both", None):
                _distances.append(memento.JSDCategoricalPoint(
                    'y', classes=n_classes, workers=config.workers))
                _sort.append('y')
            if distances in ("input", "both", None):
                _distances.append(memento.JSDCategoricalDistribution(
                    'yhat', workers=config.workers))
                _sort.append('yhat')
            _sort.append('x')  # Least importance, basically unused.
        else:
            _distances = distances
            _sort = sort
        super().__init__(
            *args, distances=_distances, batching_sort=_sort, **kwargs)

    def predict(self, x):
        """Softmax prediction of workload probabilities."""
        # tensorflow import is quite slow.
        # pylint: disable=import-outside-toplevel
        from .models import predict_workload
        return predict_workload(self.predictor, x)


class WorkloadMemory(_Ns3MultiMemory):
    """Multi-memory for the workload model."""
    coverage_mem_class = WorkloadMemoryCoverage


class TranstimeMemoryCoverage(memento.Memento):
    """Replay memory for the transtime model.

    Both ground truth and predictions are point values, and we use a fixed
    support grid on which a (continuous) probability distribution can be
    estimated.
    """

    def __init__(self, *args, distances=None, kde_bw=None, sort=None,
                 kde_support=None, config=None, **kwargs):
        config = Ns3ExperimentConfig.with_updates(config)
        if distances in ("input", "output", "both", None):
            if kde_bw is None:
                kde_bw = config.transtime_kde_bw
            if kde_support is None:
                kde_support = config.transtime_support
            _distances = []
            _sort = []
            if distances in ("output", "both", None):
                _distances.append(memento.JSDContinuousPoint(
                    'y',
                    bw=kde_bw, support=kde_support, workers=config.workers
                ))
                _sort.append('y')
            if distances in ("input", "both", None):
                _distances.append(memento.JSDContinuousPoint(
                    'yhat',
                    bw=kde_bw, support=kde_support, workers=config.workers
                ))
                _sort.append('yhat')
            _sort.append('x')  # Least importance, basically unused.
        else:
            _distances = distances
            _sort = sort
        super().__init__(
            *args, distances=_distances, batching_sort=_sort, **kwargs)


class TranstimeMemory(_Ns3MultiMemory):
    """Multi-memory for the transtime model."""
    coverage_mem_class = TranstimeMemoryCoverage


class WorkloadLARS(memento.alternatives.LossAwareBalancedReservoir):
    """Loss-aware balanced reservoir memory."""

    def classes(self, data):
        """Ground truth is class labels."""
        return data['y']

    def losses(self, data):
        """Use predictor to compute loss."""
        if (self.predictor is None) and not self.require_predictor:
            return super().losses(data)
        return [self.predictor.loss(y, yhat).numpy().item()
                for y, yhat in zip(data['y'], data['yhat'])]

    def predict(self, x):
        """Softmax prediction of workload probabilities."""
        # pylint: disable=import-outside-toplevel
        from .models import predict_workload
        return predict_workload(self.predictor, x)
