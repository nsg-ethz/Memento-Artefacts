"""Memory evaluation via data replay."""

import logging
from textwrap import dedent
from typing import Any, Dict, List, Optional

import experiment_helpers as eh
import numpy as np
import pandas as pd
from experiment_helpers.data import Path

from memento.utils import Random

from ... import replay_helper
from ..config import PufferExperimentConfig
from . import memory, models
from .data import load_inout


# Notation: the `*` syntax means that all arguments are keyword arguments.
def data_overview(*, startday, endday, model_index=0, config=None):
    """Compute general data stats."""
    config = PufferExperimentConfig.with_updates(config)
    # Check which days have useable data for replay and get paths to frames.
    # (Frame are quicker to load than the inout data).
    days = _get_valid_days(startday, endday, model_index, config)

    stats = []
    for _index, _day in enumerate(days, 1):
        # Only transmitted chunks are relevant for training Fugu.
        frame = pd.read_csv(
            config.get_data_directory(_day) / config.video_filename
        ).dropna(subset="trans_time")
        stats.append(dict(
            day=pd.to_datetime(
                frame['sent time (ns GMT)'].min(), unit='ns').date(),
            iteration=_index,
            sessions=frame['session'].nunique(),
            chunks=len(frame),
            stream_seconds=len(frame) * config.chunk_playtime,
        ))

    statframe = pd.DataFrame(stats)
    eh.data.to_csv(statframe, "results.csv.gz")
    return statframe


# Notation: the `*` syntax means that all arguments are keyword arguments.
def evaluate(*, startday, endday, replaycls, memorycls,
             config=None,
             config_overrides: Optional[Dict[str, Any]] = None,
             model_index=0):
    """Replay data from startday to endday and evaluate performance.

    Use config_updates to e.g. override train settings.
    """
    config = PufferExperimentConfig.with_updates(config, config_overrides)
    input_days = _get_valid_days(startday, endday, model_index, config)

    try:
        mem = memorycls(workers=config.workers)
    except TypeError:
        # Not all memories use multiprocessing.
        mem = memorycls()

    replay = replaycls(
        input_days,
        memory=mem,
        config=config,
        index=model_index,
        out='.',  # Saves results automatically.
    )
    replay.run()


def _get_valid_days(start, end, index, config: PufferExperimentConfig):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    logging.debug("Checking data between `%s` and `%s`,",
                  config.daystr(start), config.daystr(end))
    valid_days = []
    invalid = []
    all_days = pd.date_range(start, end, freq='D')
    for day in all_days:
        daystr = config.daystr(day)
        daydir = config.get_data_directory(day)
        inout_file = daydir / config.inout_filenames[index]
        data_file = daydir / config.video_filename
        if not (inout_file.is_file() or data_file.is_file()):
            invalid.append(f"{daystr}: missing inout/video chunk data.")
        else:
            valid_days.append(day)

    if invalid:
        logging.warning("Missing data for `%i` day(s).", len(invalid))
        logging.debug("\n".join(invalid))

    return valid_days


class PufferDataReplay(Random, replay_helper.ReplayFramework):
    """Data replay utility subclassed for Puffer.

    That is, the data loading, train, and prediction methods are implemented
    to work with the ns-3 data and models.

    Running everything, including storing models, data, results, ..., is done
    by the DataReplay class already.

    Requires a list of input files in Puffer inout format.
    The final day will only be used for eval, not training.

    retrain_from may be None (starting from scratch), "last" (starting from
    last model) or a path to a model to start from, e.g. fugu-feb

    If `jtt_upscale` is not None but a weight (usually >1), use the
    "Just train twice" strategy: https://proceedings.mlr.press/v139/liu21f.html
    - Train the model once.
    - Determine which samples were predicted wrongly.
    - Train again, scaling up the loss of the wrongly predicted samples.
    """
    # Implementation note: naming this file checkpoint*, it make sure that it gets cleaned up with other checkpoints if --force is used (just a detail).
    META_FILENAME = "checkpoint_meta.pickle.gz"

    # Allow loss as an additional metric.
    ALLOWED_TRAIN_METRICS = [
        *replay_helper.ReplayFramework.ALLOWED_TRAIN_METRICS,
        "loss", "tail_loss"
    ]

    def __init__(self, input_days, *args, index=0,
                 out=".", retrain_from=None,
                 multi_mem_filter=None, aggregate_eval=True,
                 config=None,
                 freeze_after: Optional[int] = None,
                 jtt_upscale: Optional[float] = None,
                 load_stalls: bool = False,
                 stall_upscale: Optional[float] = None,
                 matchmaker_predictors: Optional[int] = None,
                 **kwargs):
        self.config = PufferExperimentConfig.with_updates(config)
        self.index = index  # The model index, by default 0

        # Which model to retrain from (speed up training).
        if retrain_from == "fugu_feb":
            retrain_from = Path(self.config.fugu_feb_dir)
        elif retrain_from not in (None, "last"):
            retrain_from = Path(retrain_from)  # Ensure it's a valid path.
        self.retrain_from = retrain_from

        if out is None:
            raise ValueError("Puffer retraining needs an output dir!")
        self.metafile = Path(out) / self.META_FILENAME
        if multi_mem_filter is None:
            multi_mem_filter = [0]  # Replay memory comes first in MultiMem.

        if aggregate_eval and isinstance(aggregate_eval, bool):
            # If aggregate_eval is literally "True", use config values.
            self.aggregate_eval = self.config.analysis_quantiles
        else:
            # Otherwise use False or whatever values are provided.
            self.aggregate_eval = aggregate_eval

        # Past models for MatchMaker-like predictions.
        self.matchmaker_predictors = matchmaker_predictors
        self.last_predictorpaths: List[str] = []

        # Upscale factor for JTT-like training.
        self.jtt_upscale = jtt_upscale

        # Load stall information either for upscaling or selection.
        self.stall_upscale = stall_upscale
        self.load_stalls = load_stalls or (stall_upscale is not None)

        assert not (self.jtt_upscale and self.stall_upscale), \
            "Cannot use both JTT and stall upscaling."

        # Everything else is ok, preprocess inputs and save to file.
        # Eval data is data from next day.
        self.iterdays = list(zip(input_days, input_days[1:]))

        # Indicators to stop memory updates/training at some point to evaluate
        # density/performance on a fixed memory/model.
        self.freeze_after = freeze_after
        self.frozen = False

        super().__init__(*args, out=out,
                         multi_mem_filter=multi_mem_filter, **kwargs)

    def save_state(self, iteration):
        """Save additional metadata checkpoint."""
        super().save_state(iteration)  # Normal checkpooint
        logging.debug("Saving run metadata.")
        eh.data.to_pickle(self.iterdays, self.metafile)

    def load_state(self):
        """Load state and assert that metadata matches."""
        super().load_state()
        # Check metadata.
        try:
            existing_meta = eh.data.read_pickle(self.metafile)
        except FileNotFoundError:
            return  # No meta to load.
        same_inout = self._same_days(existing_meta, self.iterdays)
        if not same_inout:
            msg = "`%s` already exists and does not match input!"
            logging.critical(msg, self.metafile)
            logging.debug("Input: %s", self.iterdays)
            logging.debug("Found: %s", existing_meta)
            raise RuntimeError(msg % self.metafile)

    def remove_checkpoint(self):
        """Clean up extra metafile."""
        super().remove_checkpoint()
        self.metafile.unlink()  # Clean up.

    @staticmethod
    def _same_days(days_a, days_b):
        names_a = [day for days in days_a for day in days]
        names_b = [day for days in days_b for day in days]
        return names_a == names_b

    def load_data(self, starting_iteration=0):
        """Load inout data of day as input, and path of next eval data.

        We don't load the eval data already because it takes a lot of space
        in memory.
        """
        for day, eval_day in self.iterdays[starting_iteration:]:
            logging.debug(dedent(f"""
                Data for next iteration:
                Inout[{self.index}]:      `{day}`.
                Eval-Inout[{self.index}]: `{eval_day}`.
                """).strip())

            input_data = _load_inout(day, self.index, self.config,
                                     load_stalls=self.load_stalls)

            yield (input_data, eval_day)
            del input_data  # Allow to garbage collect.

    def total_iterations(self):
        return len(self.iterdays)

    def evaluate(self, predictor, eval_data):
        """Evaluate models."""
        eval_day = eval_data  # data is just the day, we load on demand.
        # Don't store whole inout, it can be huge.
        eval_inout = {
            self.index: load_inout(eval_day, self.index, self.config)
        }
        results = []

        # Basic eval.
        if self.matchmaker_predictors is None:
            results.append(self._eval(predictor, eval_inout))
        else:
            results.append(self._eval_matchmaker(eval_inout))

        # Additionally eval for fugu models for comparison.
        # Fugu only if available (not the case after 2022-10-06).
        fugupath = (self.config.get_data_directory(eval_day) /
                    self.config.model_pathname)
        if fugupath.is_dir():
            logging.debug("Evaluating Fugu model (`%s`).", fugupath)
            results.append(
                self._eval(fugupath, eval_inout).add_prefix("fugu_"))
        else:
            logging.debug("No Fugu model available for `%s`.", eval_day)

        febpath = self.config.fugu_feb_dir
        logging.debug("Evaluating Fugu_Feb (`%s`).", febpath)
        results.append(self._eval(febpath, eval_inout).add_prefix("fugu_feb_"))

        # Combine results.
        return pd.concat(results, axis=1)

    def _eval(self, predictor, eval_inout):
        """Use a single Fugu model to make a prediction and evaluate it."""
        predictions = models.predict(
            eval_inout, predictor, indices=[self.index],
            workers=self.config.workers)

        return models.evaluate(
            predictions, predictor, self.index, aggregate=self.aggregate_eval)

    def _eval_matchmaker(self, eval_inout):
        """Evaluate "MatchMaker"-style, using the best of the last n models.

        Concretely, MatchMacker uses an ensemble to make predictions, and
        uses proxy metrics to decide for each training sample which model might
        give the best prediction. This model is that actually used to predict.

        It keeps models for the past n units of data; in our case a unit of
        data is one day.

        We use an idealized version of MatchMaker: We keep the last n models,
        predict with all of them, and use the model with the best prediction.
        That is, we pretend that MatchMaker made the best possible decision.

        Paper: https://proceedings.mlsys.org/paper/2022/file/1c383cd30b7c298ab50293adfecb7b18-Paper.pdf
        """
        # Make predictions for all models and pick best.
        predictions = [
            models.predict(eval_inout, _path, indices=[self.index],
                           workers=self.config.workers)
            for _path in self.last_predictorpaths
        ]

        # For our eval, the best model is the one with the highest probability
        # for the true event.
        key = (str(self.index), "p")
        merged = predictions[0]
        for other in predictions[1:]:
            # Quote docs: Where cond is True, keep the original value. Where
            # False, replace with corresponding value from other.
            keep_current = merged[key] >= other[key]
            merged = merged.where(keep_current, other)

        # Finally, evaluate best predictions
        # Note: We only need predictor to get structural info, we can use any.
        return models.evaluate(
            merged, self.predictor, self.index, aggregate=self.aggregate_eval)

    def eval_driftsurf(self, predictor, eval_inout):
        """Compare Fugu vs FuguFeb."""
        raise NotImplementedError()

    def train(self, traindata, last_predictor=None):
        """Training with different reference models."""
        if self.retrain_from is None:
            ref = None
        elif self.retrain_from == "last":
            ref = last_predictor
        else:
            ref = self.retrain_from

        if self.jtt_upscale:
            # Train and store the first model.
            jtt_path = self.predictorpath / "jtt_model"
            models.train(traindata['x'], traindata['y'], jtt_path,
                         weights=None, index=self.index, refmodeldir=ref,
                         config=self.config)
            # Load model, make predictions
            jtt_model = self._load_predictor(jtt_path)
            correct_label = models.predict_and_compare(
                jtt_model, traindata['x'], traindata['y'])
            # np.where(condition, a, b) returns a where condition else b
            # Correct labels get weigth 1.0.
            weights = np.where(correct_label, 1.0, self.jtt_upscale)
        elif self.stall_upscale:
            # Weights are 1.0 for non-stalled, and stall_upscale for stalled.
            weights = np.where(traindata['stalled'], self.stall_upscale, 1.0)
        else:
            weights = None

        models.train(traindata['x'], traindata['y'], self.predictorpath,
                     weights=weights, index=self.index, refmodeldir=ref,
                     config=self.config)

        if self.matchmaker_predictors is not None:
            # If we use matchmaker-style predictions, remember the last models.
            # First load all models, then update list, then save them again.
            loaded = [self._load_predictor(mpath)
                      for mpath in self.last_predictorpaths]
            loaded.append(self._load_predictor(self.predictorpath))
            loaded = loaded[-self.matchmaker_predictors:]  # Max length.
            self.last_predictorpaths = []
            for i, model in enumerate(loaded):
                _path = self.predictorpath / f"ensemble/{i}"
                self.last_predictorpaths.append(_path)
                self._save_predictor(model, _path)

        # We return the directory around for compatibility with other methods.
        return self.predictorpath

    def compute_stats(self, iteration, memory_update_time, mem_data, eval_day):
        """If "frozen", compute special density stats."""
        retrain, stats = super().compute_stats(
            iteration, memory_update_time, mem_data, eval_day)

        # If frozen, compute special density stats.
        if self.frozen:  # Never retrain if frozen.
            # Important: in the Puffer replay, we only load eval data on demand.
            eval_inout = load_inout(eval_day, self.index, self.config)
            eval_data = {'x': eval_inout['in'], 'y': eval_inout['out']}
            del eval_inout

            # Should I make this parametrizable? I only really need it for
            # one experiment. So for now not.
            for batchsize in [256, 1024]:
                _mem = memory.PufferMemento(
                    size=None,  # Infinite
                    batching_size=batchsize,
                    workers=self.config.workers,
                )
                _mem.update_predictor(self.predictor)

                # Bit hacky:
                # Run data through memory just for the preprocessing steps.
                _mem.insert(eval_data)
                _data_new = _mem.get()
                _mem.data = {}  # Reset, then do other data.
                _mem.insert(self.memory.get())
                _data_mem = _mem.get()
                _mem.data = {}  # Reset again to free memory.

                _data_new = _mem.batch(_data_new)
                _data_mem = _mem.batch(_data_mem)

                # Distances.
                _d = _mem.compute_matrices(_data_mem, _data_new)

                # Densities.
                def _density(mem, distances, axis):
                    # Also return the sorted densities to analyse memory
                    # contents. Implementation detail: list(tuple) for
                    # compatibility with pandas and csv storage.
                    return [tuple(
                        mem.agg([d.mean(axis=axis) for d in distances])
                    )]

                # Add to stats.
                stats[f'density_mem_new_{batchsize}'] = _density(_mem, _d, 1)
                stats[f'density_new_mem_{batchsize}'] = _density(_mem, _d, 0)

        return retrain, stats

    def retrain_needed(self, _data):
        """Check for increase in model loss if specified."""
        if self.frozen:  # Never retrain if frozen.
            return False, {}

        if self.train_metric == "loss":
            if self.predictor is None:
                return True, {}  # No comparison, need to train in any case.

            # Load the actual model.
            loaded_predictor = self._load_predictor(self.predictor)
            last_train_loss = models.predict_loss(
                loaded_predictor, self.last_train_data['x'], self.last_train_data['y'])
            current_loss = models.predict_loss(
                loaded_predictor, _data['x'], _data['y'])

            loss_change = (current_loss - last_train_loss)
            relative_change = loss_change / last_train_loss
            significant_increase = relative_change >= self.train_threshold

            return significant_increase, {
                'last_train_loss': last_train_loss,
                'current_loss': current_loss,
                'loss_change': loss_change,
                'loss_change_rel': relative_change,
            }
        else:
            return super().retrain_needed(_data)

    def _sample_diff_filter(self, _data):
        """Only consider x and y when comparing samples."""
        return {'x': _data['x'], 'y': _data['y']} if _data else {}

    def _update_memory(self, iteration, _data):
        """Only update memory if not frozen, otherwise ignore all data."""
        # Check if we need to freeze. (Note: iteration is zero-based, thus >=)
        if (self.freeze_after is not None) and (iteration >= self.freeze_after):
            self.frozen = True

        if not self.frozen:
            super()._update_memory(iteration, _data)

    def save_predictor(self, predictor, path):
        """Predictor is already a path.

        Most of the time we don't need to actually save. If we do, use method
        below.
        """
        return

    def load_predictor(self, path):
        """Predictor is already a path.

        Most of the time we don't need to actually load. If we do, use method
        below.
        """
        return path

    def _save_predictor(self, predictor, path):
        """If we _really_ need to save the model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        predictor.save(path / f"py-{self.index}.pt")

    def _load_predictor(self, path):
        """If we _really_ need to load the model."""
        return models.load_model(Path(path) / f"py-{self.index}.pt")

    def predictor_exists(self, path):
        """The directory may contain checkpoints etc, so check precisely."""
        predictorfile = path / f"py-{self.index}.pt"
        return predictorfile.is_file()


def _load_inout(day, index, config: PufferExperimentConfig, load_stalls=False):
    inout = load_inout(day, index, config)
    # Extra info for analysis.
    label = pd.to_datetime(
        np.min(inout['ts']), unit='ns').date().strftime(config.datefmt)
    loaded = dict(
        x=inout['in'],
        y=inout['out'],
        label=np.repeat(label, len(inout['in'])),
    )

    if load_stalls:
        is_stalled = _load_session_stalls(day, config)
        loaded['stalled'] = np.array([
            is_stalled[s] for s in inout['session']])

    return loaded


def _load_session_stalls(day, config: PufferExperimentConfig):
    """Return dict of sessions and whether they are stalled.

    That is, session with any video chunk that has cum_rebuf > 0.
    """
    video_file = config.get_data_directory(day) / config.video_filename
    video_data = pd.read_csv(video_file)
    return video_data.groupby("session")["cum_rebuf"].any().to_dict()
