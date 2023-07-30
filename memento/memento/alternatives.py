"""Alternative replay memory algorithms.

These algorithms exists mainly to benchmark the main replay memory, which is
implemented using kernel-density based coverage maximization.
However, they might be interesting as baselines for other algorithms or
research as well.
"""

import logging
from collections import defaultdict

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import softmax

from . import bases, utils

logger = logging.getLogger("memory.memory")


class SampleMetricMemory(utils.Random, bases.MemoryBase):
    """Instead of density, select samples based on alternative metrics.

    The higher the metric, the more likely a sample is forgotten.
    """

    def __init__(self, *args,
                 temperature: float = 0.01,
                 batching_agg=np.mean,
                 **kwargs):
        self.temperature = temperature
        self.batching_agg = batching_agg
        super().__init__(*args, **kwargs)

    def metric(self, data) -> ArrayLike:
        """Return the metric for each sample."""
        return np.ones(utils.sample_count(data))  # Dummy: uniform.

    def select(self, max_samples, current_data, new_data):
        """Keep samples with low metric. High metric = high discard prob."""
        # Make no distinction between new and old samples.
        data = utils.merge_datadicts([current_data, new_data])
        total = utils.sample_count(data)
        logger.debug("Removing `%i` of `%i` samples.",
                     total - max_samples, total)
        data['metric'] = self.metric(data)

        # Corner case: If the data is too skewed or temperature too high, there
        # many discard probabilities will be 0 and there may be less non-zero
        # probabilities left than samples we need to keep. Discard all samples
        # with non-zero probability and recompute probabilities.
        while utils.sample_count(data) > max_samples:
            if self.batching_size is not None:
                # Discard per batch.
                data = np.array(self.batch(data))
                total = len(data)
                to_discard = int(np.ceil(
                    total - max_samples / self.batching_size))
                metrics = np.array([
                    self.batching_agg(d['metric']) for d in data])
            else:
                # Discard per sample.
                total = utils.sample_count(data)
                to_discard = total - max_samples
                metrics = data['metric']

            # Compute probabilities to _discard_ samples, based on metric.
            probabilities = softmax(metrics / self.temperature)

            n_nonzero = (probabilities > 0).sum()
            if n_nonzero < to_discard:
                logger.debug("Insufficient samples with non-zero probabily."
                             "removing `%i` samples and "
                             "recomputing probabilities.", n_nonzero)
                # Too few non-zero probabilities, discard all of those at least.
                # That is, keep all the samples with zero probabilities.
                keep_index = (probabilities == 0)
            else:
                # Select samples to discard based on probabilities.
                to_remove = self.rng.choice(
                    total, size=to_discard, p=probabilities, replace=False)
                keep_index = np.ones(total).astype(bool)
                keep_index[to_remove] = False  # i.e. do not keep them.

            # Remove selected elements.
            if self.batching_size is not None:
                # Per batch.
                data = utils.merge_datadicts(data[keep_index])
            else:
                # Per sample.
                data = {k: v[keep_index] for k, v in data.items()}

        # Remove metrics and return data.
        del data['metric']
        return data


class NoPredictorMemoryBase(bases.MemoryBase):
    """Turn of the predictor by default for memories that don't benefit."""

    def __init__(self, *args,
                 require_predictor=False,
                 predict_on_insert=False,
                 predict_on_update=False,
                 **kwargs):
        super().__init__(*args,
                         require_predictor=require_predictor,
                         predict_on_insert=predict_on_insert,
                         predict_on_update=predict_on_update,
                         **kwargs)


class ReservoirMemory(utils.Random, NoPredictorMemoryBase):
    """Replay memory with reservoir/global distribution matching strategy.

    This memory supports a steady forgetting rate.
    """

    def __init__(self, *args, forget: float = 0.0, key='random', **kwargs):
        """Initialize reservoir."""
        self.key = key
        self.forget = forget
        super().__init__(*args, **kwargs)

    def insert_datadict(self, datadict):
        """Draw (permanent) random keys for new samples."""
        datadict[self.key] = self.rng.random(size=utils.sample_count(datadict))
        return super().insert_datadict(datadict)

    def select(self, max_samples, current_data, new_data):
        """Select batches based on random draws."""
        if current_data and new_data and self.forget:
            # Do not forget more batches than incoming data.
            total = utils.sample_count(current_data)
            num_forget = min(int(np.round(total * self.forget)),
                             utils.sample_count(new_data))
            logger.debug("Forgetting `%i` samples.", num_forget)
            keep_index = \
                self.rng.choice(total, total - num_forget, replace=False)
            current_data = {k: v[keep_index] for k, v in current_data.items()}
        else:
            logger.debug("No forgetting.")

        merged = utils.merge_datadicts([current_data, new_data])

        # Sort on random column and select first max_samples.
        sort_ind = np.argsort(merged[self.key])
        return {k: v[sort_ind][:max_samples] for k, v in merged.items()}


class LossAwareBalancedReservoir(utils.Random, bases.MemoryBase):
    """LARS

    Source: http://arxiv.org/abs/2010.05595
    However, we needed to update the implementation to handle larger sample
    sizes.

    Subclasses need to implement `losses()` and `classes()`.
    """

    def __init__(self, *args, **kwargs):
        self.observed_samples = 0
        super().__init__(*args, **kwargs)

    def classes(self, data) -> ArrayLike:
        """Return a class label for each batch/sample."""
        return np.zeros(utils.sample_count(data))  # Dummy: same class for each

    def losses(self, data) -> ArrayLike:
        """Return loss for each batch/sample."""
        return np.ones(utils.sample_count(data))  # Dummy: same loss for each

    def select(self, max_samples, current_data, new_data):
        """Select batches based on random samples."""
        data = {k: v[:max_samples] for k, v in current_data.items()}
        remaining = max_samples - utils.sample_count(data)
        if remaining:
            logger.debug("Accepting `%i` samples immediately.", remaining)
            new_in = {k: v[:remaining] for k, v in new_data.items()}
            data = utils.merge_datadicts([data, new_in])
            new_data = {k: v[remaining:] for k, v in new_data.items()}

        # Ensure that observed batches is properly initialized.
        # (as this func is not called until capacity is reached.)
        self.observed_samples = max(self.observed_samples,
                                    utils.sample_count(data))

        len_new = utils.sample_count(new_data)
        logger.debug("Fetching random keys and checking samples.")
        _lower = self.observed_samples
        _upper = self.observed_samples + len_new
        keys = [self.rng.choice(i+1) for i in range(_lower, _upper)]

        logger.debug("Update number of observed samples.")
        self.observed_samples += len_new

        accepted_idx = [idx
                        for (idx, key) in zip(np.arange(len_new), keys)
                        if key < max_samples]
        logger.debug("`%i` samples accepted.", len(accepted_idx))
        if not accepted_idx:
            return data
        accepted = {k: v[accepted_idx] for k, v in new_data.items()}

        logger.debug("Computing losses and class count.")
        size = utils.sample_count(data)
        mem_importances = -1 * np.array(self.losses(data))
        mem_classes = np.array(self.classes(data))
        counts_per_class = defaultdict(
            int,
            zip(*np.unique(mem_classes, return_counts=True))
        )
        classes = np.array(self.classes(accepted))
        importances = -1 * np.array(self.losses(accepted))

        logger.debug("Replacing old samples.")
        # Work with index arrays, and select actual data in the end.
        combined = utils.merge_datadicts([data, accepted])  # select from here.
        len_mem = utils.sample_count(data)
        len_new = utils.sample_count(accepted)
        final_idx = np.arange(len_mem)  # originally only old data.
        new_idx = np.arange(len_new) + len_mem  # these indices will move in.

        # Avoid computing alpha from scratch each iteration
        # Note: the sum of all class counts is equal to the sum of squares.
        mean_absimportance = sum(np.abs(mem_importances)) / size
        # For each accepted sample, determine which sample it will replace.
        # Replaying them one-by-one.
        for idx, _cls, importance in zip(new_idx, classes, importances):
            mean_classcount = (  # Fairly cheap.
                sum(c**2 for c in counts_per_class.values()) / size
            )
            # Not so cheap, but no other way.
            classcounts = np.array([counts_per_class[_class]
                                    for _class in mem_classes])
            probs = self.merge_scores(
                mem_importances, classcounts,
                mean_absimportance, mean_classcount
            )
            to_replace = (
                self.rng.choice(range(len(final_idx)), p=probs, size=1).item())

            # Update state
            _imp = mem_importances[to_replace]
            mem_importances[to_replace] = importance
            mean_absimportance += ((np.abs(importance) - np.abs(_imp)) / size)
            counts_per_class[mem_classes[to_replace]] -= 1
            counts_per_class[_cls] += 1
            mem_classes[to_replace] = _cls
            final_idx[to_replace] = idx

        # Now actually select.
        return {k: v[final_idx] for k, v in combined.items()}

    def merge_scores(self, importance, counts, meanimportance, meancounts):
        """Merging scores as in the published code.

        https://github.com/hastings24/rethinking_er/blob/master/utils/buffer_tricks.py#L63-L70
        """
        alpha = meanimportance * meancounts
        scores_raw = (0.5 * importance / alpha) + (0.5 * counts)

        _min = scores_raw.min()
        _max = scores_raw.max()
        if _max != _min:
            scores_raw = (scores_raw - _min) / (_max - _min)

        return scores_raw / scores_raw.sum()


class FIFOMemory(NoPredictorMemoryBase):
    """Classical first-in-first-out memory."""

    def select(self, max_samples, current_data, new_data):
        """Select batches based on random draws."""
        # Append new data to the end (first in).
        merged = utils.merge_datadicts([current_data, new_data])
        # Keep last samples (first out).
        return {k: v[-max_samples:] for k, v in merged.items()}


class InfiniteMemory(NoPredictorMemoryBase):
    """Dummy memory class that just remembers and returns the all samples.

    Disregards any size constraints.
    """

    def select(self, max_samples, current_data, new_data):
        """Select everything disregarding size."""
        return utils.merge_datadicts([current_data, new_data])


class NoMemory(NoPredictorMemoryBase):
    """Dummy memory class that stores nothing."""

    def select(self, max_samples, current_data, new_data):
        """Select nothing."""
        return {}
