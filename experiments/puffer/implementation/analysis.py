"""Functions to evaluate the results published by the Puffer project.

This analysis focuses on additional metrics than mean SSIM and time stalled.
"""

import logging
from os import PathLike
from typing import Optional, Union

import experiment_helpers as eh
import numpy as np
import pandas as pd
from experiment_helpers.data import Path

from ..config import PufferExperimentConfig
from .data import InOut, load_inout
from .models import InoutOrFrame, predict

idx = pd.IndexSlice


# `*` syntax means that all arguments are keyword arguments.
def analyze_day(*, day, index=None, config=None):
    """Evaluate video data by abr for a given day and model."""
    config = PufferExperimentConfig.with_updates(config)
    day = pd.to_datetime(day)
    daydir = config.get_data_directory(day)
    videodatapath = daydir / config.video_filename
    logging.info("Evaluating `%s`.", videodatapath)

    try:
        logging.debug("Loading video data data from `%s`.", videodatapath)
        frame = pd.read_csv(videodatapath)
        inout = load_inout(day, index=index, config=config)

        logging.debug("Filtering ABRS.")
        frame, inout = _filter_abrs(frame, inout, config)

        logging.debug("Computing prediction scores.")
        frame = _add_scores(frame, inout, config)

        logging.debug("Computing statistics.")
        statistics = _compute_statistics(frame, config)

        # Append day info and save.
        eh.data.to_csv(statistics.assign(day=day), "results.csv.gz")
    except (AssertionError, FileNotFoundError, NotADirectoryError) as error:
        # Some days are known to have incomplete data.
        # Some days are known to have incomplete or corrupted data.
        # For those, turn error into info and return.
        if day in config.data_known_missing:
            logging.info(
                "Analysis failed for %s, but data is known as missing.",
                config.daystr(day)
            )
        else:
            raise error


# Helper functions.
# =================

def _compute_statistics(frame: pd.DataFrame, config: PufferExperimentConfig):
    # Group by stalled/unstalled
    _rebuf = frame.groupby("session")['cum_rebuf'].max()
    stalled_sessions = set(_rebuf[_rebuf > 0].index)
    stall_index = frame['session'].isin(stalled_sessions)

    selections = {
        'all': _clean_frame(frame, config),
        'stalled': _clean_frame(frame[stall_index], config),
        'unstalled': _clean_frame(frame[~stall_index], config),
    }

    return pd.concat(
        [_abr_statistics(_frame, config).assign(selection=_key)
         for _key, _frame in selections.items() if not _frame.empty],
        ignore_index=True
    )


def _filter_abrs(frame: pd.DataFrame, inout: InOut,
                 config: PufferExperimentConfig):
    filtered = (
        frame
        [frame['abr'].isin(config.evaluation_mapping)]
        .assign(abr=lambda df: pd.Categorical(
            df['abr'].map(config.evaluation_mapping),
        ))
    )
    # Remove values that don't exist anymore
    for col in filtered.select_dtypes("category"):
        filtered[col] = filtered[col].cat.remove_unused_categories()

    filtered_inout = _filter_inout(filtered, inout)

    return filtered, filtered_inout


def _abr_statistics(frame: pd.DataFrame, config: PufferExperimentConfig,
                    groupby=None):
    """Filter frame and compute statistics by abr.

    Always include statistics for SSIM, delta ssim, and stalls.
    If `logscore_0` and `logscore_all` columns are present, also compute score
    statistics.
    Finally, include session and chunk counts.
    """
    frame = _clean_frame(frame, config)
    if groupby is None:
        groupby = ["abr"]
    results = []

    results.append(
        frame
        .groupby(groupby, observed=True, group_keys=True)
        [["ssim_index"]]
        .apply(_mean_and_quantiles, config)
        .transform(ssim_to_db)
        .reset_index()
        .rename(columns={"ssim_index": "value"})
        .assign(variable="ssim")
    )

    results.append(
        frame
        .groupby(groupby, observed=True, group_keys=True)
        .apply(lambda df: _mean_and_quantiles(
            delta_ssim_db(df)[["delta_ssim"]], config,
        ))
        .reset_index()
        .rename(columns={"delta_ssim": "value"})
        .assign(variable="delta_ssim")
    )

    results.append(
        frame
        .groupby(groupby, observed=True, group_keys=True)
        .apply(_stallstats, config)
        .droplevel(-1)  # superflous index because apply returns frames.
        .reset_index()
        .assign(variable="stream")
    )

    for metric in [
        "y", "y_hat",
        "logscore", "logscore_0",
        "error", "error_0",
        "error_rel", "error_rel_0"
    ]:
        try:
            results.append(
                frame
                .dropna(subset=[metric])  # logscore_all only after 5th chunk.
                .groupby(groupby, observed=True, group_keys=True)
                [[metric]]
                .apply(_mean_and_quantiles, config)
                .reset_index()
                .rename(columns={metric: "value"})
                .assign(variable=metric)
            )
        except KeyError:
            pass  # Ok if missing.

    # Also add number of chunks and sessions per group
    results.append(
        frame
        .groupby(groupby, observed=True, as_index=False)
        .size()
        .rename(columns={"size": "value"})
        .assign(metric="count", variable="chunk")
    )
    results.append(
        frame
        .groupby(groupby, observed=True, as_index=False)
        [['session']]
        .nunique()
        .rename(columns={"session": "value"})
        .assign(metric="count", variable="session")
    )

    return (
        pd
        .concat(results, ignore_index=True)
        .astype({'metric': 'category', 'variable': 'category'})
        .astype({key: 'category' for key in groupby})
    )


def _mean_and_quantiles(frame: pd.DataFrame, config: PufferExperimentConfig):
    """Aggregate mean, std, root-mean-square (rms), and quantiles."""
    aggregate_basic = frame.agg(['mean', 'std'])
    aggregate_rms = np.sqrt((frame ** 2).mean().to_frame(name="rms").T)
    aggregate_quantiles = frame.quantile(config.analysis_quantiles)
    return (
        pd
        .concat([aggregate_basic, aggregate_rms, aggregate_quantiles])
        .rename_axis(index="metric")
        # .reset_index()
    )


def _stallstats(frame: pd.DataFrame, config: PufferExperimentConfig):
    """Compute summarized stall statistics."""
    # Play and rebuffering time per session.
    grouped = frame.groupby(["session"], observed=True)
    playtime = grouped.size() * config.chunk_playtime  # n_chunks * duration.
    rebuftime = grouped['cum_rebuf'].max()  # Already cumulative.

    # Aggregate across all sessions.
    total_play = playtime.sum()
    total_buf = rebuftime.sum()
    total_stream = total_play + total_buf
    stalled = 100 * total_buf / total_stream
    return pd.DataFrame([
        {"metric": "playtime", "value": total_play},
        {"metric": "rebuftime", "value": total_buf},
        {"metric": "streamtime", "value": total_stream},
        {"metric": "stalled", "value": stalled},
    ])


# Add predictions and compute predictions scores.
# ===============================================

def _add_scores(frame: pd.DataFrame, inout: Optional[InOut],
                config: PufferExperimentConfig):
    """Return a new frame with added prediction scores."""
    day = pd.to_datetime(frame[time_col(frame)].min(), unit='ns')
    scores = []

    logging.debug("Scores for Fugu (if available).")
    try:
        # Note: The fugu model published on day X is used for data on day X.
        fugu_inout = _filter_inout(frame, inout, "fugu")
        dir_fugu = config.get_data_directory(day) / config.model_pathname
        scores.append(_get_scores(fugu_inout, dir_fugu,
                                  workers=config.workers))
    except (AssertionError, FileNotFoundError):
        pass  # Discontinued at some point, but missing on other days before.

    logging.debug("Scores for FuguFeb.")
    fugufeb_inout = _filter_inout(frame, inout, abr="fugu-feb")
    scores.append(_get_scores(fugufeb_inout, config.fugu_feb_dir,
                              workers=config.workers))

    logging.debug("Scores for Memento-trained models (if available).")
    for abr in frame['abr'].unique():
        _dir = config.memento_model_directories.get(abr)
        if not _dir:
            continue  # This is only for ABRs trained by us.
        _models = _get_memento_models(_dir, day, config)
        if not _models or not all(_m.exists() for _m in _models):
            logging.warning("No model data for `%s` found in `%s`!",
                            abr, _dir)
            continue

        _inout = _filter_inout(frame, inout, abr)
        scores.append(_get_scores(_inout, _models, workers=config.workers))

    return frame.join(pd.concat(scores), on=scores[0].index.names)


def _get_scores(input_data: InoutOrFrame,
                modelpath: Union[PathLike, str],
                workers: int = 1):
    predictions = predict(input_data, modelpath, workers=workers)

    times = predictions.loc[:, idx[:, ['y', 'y_hat']]]

    logscores = (
        predictions
        .loc[:, idx[:, 'p']]
        .apply(np.log)  # Logscore.
        .rename(columns={'p': 'logscore'})
    )

    _a = predictions.loc[:, idx[:, ['y']]].rename(columns={'y': 'error'})
    _p = predictions.loc[:, idx[:, ['y_hat']]].rename(
        columns={'y_hat': 'error'})
    errors = _p - _a   # Error = Prediction - Actual.
    errors_rel = (errors / _a).rename(columns={'error': 'error_rel'})

    metrics = pd.concat([times, logscores, errors, errors_rel], axis=1)
    first = (
        metrics
        .loc[:, idx['0', :]]
        .droplevel(0, axis=1)
        .add_suffix("_0")
    )
    mean = metrics.groupby(level=1, axis=1).mean()

    return pd.concat([mean, first], axis=1)


def _get_memento_models(basedir, day, config: PufferExperimentConfig):
    """Find the memento model in use on `day`.

    We cannot guarantee that this model was used, as we have no feedback on
    when it is actually pulled -- so there may be a mismatch of about one day.
    """
    try:
        return [_get_last_model(index, basedir, day, config)
                for index in range(5)]
    except FileNotFoundError:
        return []


def _get_last_model(index, basedir, day, config: PufferExperimentConfig):
    """Get model for a single day and index."""
    daystr = config.daystr(day)
    statsfile = Path(basedir) / f"{index}/{daystr}/stats_{index}.json.gz"
    if statsfile.exists():
        modeldir = str(eh.data.read_json(statsfile)['model_dir'])
    else:
        # Try old format.
        statsfile = Path(basedir) / f"{index}/{daystr}/stats_{index}.pickle.gz"
        modeldir = str(eh.data.read_pickle(statsfile)['model_dir'])

    # If configured, update path (e.g. to map old paths to new locations).
    for old, new in config.memento_model_directory_replacements.items():
        modeldir = modeldir.replace(old, new)

    return Path(modeldir) / f"py-{index}.pt"


# Other helpers to process frames.
# ================================

def _filter_inout(frame: pd.DataFrame, inout: InOut, abr: str = None):
    """Return only subset of inout for provided sessions.

    If abr is provided, select sessions in frame for this abr,
    otherwise all sessions in frame.
    """
    if abr is None:
        sessions = frame['session'].unique()
    else:
        assert abr in frame['abr'].unique(), "ABR does not exist!"
        sessions = frame.loc[frame['abr'] == abr, 'session'].unique()

    filtered = []
    for inout_data in inout:
        # Pandas is very efficient at this, using a categorical series beats
        # np.isin by orders of magnitude!
        index = (
            pd.Series(inout_data['session'], dtype='category')
            .isin(sessions)
            .to_numpy()
        )
        filtered.append({
            key: np.array(value)[index]
            for key, value in inout_data.items()
        })
    return filtered


def _clean_frame(frame: pd.DataFrame, config: PufferExperimentConfig):
    """Filter dataframe to keep same sessions as Puffer does.

    Concretely:
    - Remove all chunks that were not acknowledged.
    - Remove all sessions with less than 4s acked video playtime.
    """
    def _cat_cleanup(column: str):
        def _clean_col(dataframe: pd.DataFrame):
            try:
                return dataframe[column].cat.remove_unused_categories()
            except AttributeError:
                return dataframe[column]  # Not categorical.
        return _clean_col

    return (
        frame
        # Remove all chunks without ack, i.e. without trans time
        .dropna(subset=["trans_time"])
        .groupby("session", observed=True)
        .filter(lambda group: len(group) >= config.analysis_min_chunks)
        # Cleanup: Remove empty categories.
        .assign(
            session=_cat_cleanup('session'),
            abr=_cat_cleanup("abr"),
            cc=_cat_cleanup("cc"),
        )
    )


# Conversion between SSIM in (0, 1) scale and db scale as done by Puffer.
# =======================================================================

def ssim_to_db(_data):
    """Convert ssim to db as done by Puffer."""
    with np.errstate(divide="ignore"):
        return -10 * np.log10(1 - _data)


def delta_ssim_db(frame):
    """Compute the ssim delta (in dB space)."""
    with pd.option_context('mode.use_inf_as_na', True):
        deltas = (
            frame
            .sort_values(by=[time_col(frame)], ascending=True)
            .groupby("session", observed=True)
            [["ssim_index"]]
            .apply(lambda df: (
                df
                .transform(ssim_to_db)
                .diff()
                .abs()
                .dropna()
            ))
            .droplevel(0)
            .rename(columns={"ssim_index": "delta_ssim"})
        )
    return frame.assign(delta_ssim=deltas)


def time_col(frame):
    """Get the name of the time col, ensuring backwards compatibility."""
    if "sent time (ns GMT)" in frame.columns:
        return "sent time (ns GMT)"
    if "time (ns GMT)" in frame.columns:
        return "time (ns GMT)"
    raise ValueError("No time column found.")