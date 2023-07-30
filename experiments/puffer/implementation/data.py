"""Functions to download and format puffer data."""
# pylint: disable=import-outside-toplevel, redefined-builtin

import csv
import gc
import json
import logging
import tarfile
import time
import urllib
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from tempfile import TemporaryDirectory

import experiment_helpers as eh
import numpy as np
import pandas as pd
import requests
from experiment_helpers.data import Path, open
from experiment_helpers.typing import ArrayLike, Dict, List, Union, cast

from ..config import PufferExperimentConfig

InOutDict = Dict[str, ArrayLike]
InOut = List[InOutDict]

DownloadErrors = (
    urllib.error.HTTPError, AssertionError, tarfile.TarError
)


# Download functions.
# ===================

# `*` syntax means that all arguments are keyword arguments.

def download_data(*, day, config=None):
    """Download video data and prepare inout data.

    If wait is truthy, don't fail if the data is not available, _if_ the data
    is from the previous day. Instead, wait for the specified amount of time
    and then try again. `wait` can be anything accepted by `pd.Timedelta`.
    If the data is aside from yesterday, it does not make sense to wait,
    as the data will not be published today.

    If require_model is false, no error will be raised if no model cannot be
    downloaded.

    Because we require the model inout data (for the first model) for most
    experiments, we precompute it. The other inout data can be computed on
    demand.
    """
    config = PufferExperimentConfig.with_updates(config)
    if day == "fugufeb":
        day = config.fugu_feb_day
    else:
        day = pd.to_datetime(day)
    day = cast(datetime, day)  # Type hint.
    if config.download_poll_interval is not None:
        wait = pd.Timedelta(config.download_poll_interval).total_seconds()
    else:
        wait = False
    daydir = config.get_data_directory(day)
    videofile = daydir / config.video_filename
    modeldir = daydir / config.model_pathname

    try:
        # Video data.
        while True:
            try:
                if not videofile.is_file():
                    logging.debug("Downloading video data to `%s`.", videofile)
                    df_video = download_video_sessions(day, config)
                    eh.data.to_csv(df_video, videofile)
                else:
                    logging.debug("Video data exists in `%s`.", videofile)
                break
            except DownloadErrors as error:
                yesterday = (pd.to_datetime("today")
                             - pd.Timedelta(days=1)).date()
                if day.date() >= yesterday:
                    # The data may not be available yet.
                    if wait:
                        logging.debug(
                            "Recent data not available yet, sleeping.")
                        time.sleep(wait)
                    else:
                        logging.warning("Recent data not available yet.")
                raise error

        # Model data.
        if not modeldir.is_dir():
            logging.debug("Downloading model data to `%s`.", modeldir)
            try:
                download_model_data(day, modeldir, config)
            except DownloadErrors as error:
                if day > config.last_fugu_day:
                    logging.debug("No model data available after `%s`.",
                                  config.daystr(config.last_fugu_day))
                else:
                    raise error
        else:
            logging.debug("Model data exists in `%s`.", modeldir)
    except DownloadErrors as error:
        # Some days are known to have incomplete or corrupted data.
        # For those, turn error into info and return.
        if day in config.data_known_missing:
            logging.info(
                "Download failed for `%s`, but it is known as missing.",
                config.daystr(day)
            )
        else:
            raise error


def preprocess_data(*, day, config=None):
    """Precompute inout data for a given day.

    Speeds up loading data at the expense of disk space.
    """
    config = PufferExperimentConfig.with_updates(config)
    if day == "fugufeb":
        day = config.fugu_feb_day
    else:
        day = pd.to_datetime(day)
    day = cast(datetime, day)  # Type hint.

    daydir = config.get_data_directory(day)
    videofile = daydir / config.video_filename
    inoutfiles = [daydir / filename for filename in config.inout_filenames]

    if all(inoutfile.is_file() for inoutfile in inoutfiles):
        logging.debug("Inout data exists in `%s`.", daydir)
    else:
        logging.debug("Loading video data from `%s`.", videofile)
        try:
            df_video = eh.data.read_csv(videofile)
        except FileNotFoundError as error:
            yesterday = (pd.to_datetime("today") - pd.Timedelta(days=1)).date()
            if day in config.data_known_missing:
                logging.info(
                    "No data for `%s`, but it is known as missing.",
                    config.daystr(day)
                )
            else:
                if day.date() >= yesterday:
                    logging.warning(
                        "No data for `%s`, but it may not be released yet.",
                        config.daystr(day)
                    )
                raise error
        else:
            logging.debug("Preprocessing inout data.")
            all_inout = to_ttp_inout(df_video)

            for inout, inoutfile in zip(all_inout, inoutfiles):
                logging.debug("Writing inout data to `%s`.", inoutfile)
                eh.data.to_npz(inout, inoutfile)


def download_experiments(url, config: PufferExperimentConfig):
    """Load the mapping from experiment `id` to `abr` and `cc`.

    Note: more information is in the data (like resolutions
    and database config), but we drop everything we don't need.
    """
    response = requests.get(url, timeout=config.download_timeout)
    response.raise_for_status()
    # Each line contains one experiment
    experiments = {}
    for line in response.text.splitlines():
        expt_id, expt_data = line.split(" ", 1)
        expt_data = json.loads(expt_data)
        # The key identifying the abr algorith can vary in early datasets.
        abr = (expt_data.get("abr_name") or
               expt_data.get("abr_algorithm") or
               expt_data.get("abr"))

        experiments[int(expt_id)] = dict(
            abr=abr,
            cc=expt_data.get('cc'),
        )
    return pd.DataFrame.from_dict(experiments, orient='index')


def download_video_sessions(day, config: PufferExperimentConfig):
    """Download video session data (video chunks).

    Load sent, acked, and experiment data and merge them into a single
    dataframe. As described in the puffer data description, we use the session
    id and video timestamp to join dataframes.
    """
    sent_url = config.get_puffer_data_url(day, "video_sent")
    acked_url = config.get_puffer_data_url(day, "video_acked")
    exp_url = config.get_puffer_data_url(day, "exp")

    # pandas can read directly over the web, but experiments have a proprietary
    # format that we need to parse manually
    sent_frame = pd.read_csv(sent_url)
    acked_frame = pd.read_csv(acked_url)
    exp_frame = download_experiments(exp_url, config)

    for frame in [sent_frame, acked_frame, exp_frame]:
        # For some days, e.g. 2022-02-01, the file exists, but contains no data.
        assert not frame.empty

    index = ["session_id", "index", "video_ts"]
    acked_preprocessed = acked_frame.set_index(index)[["time (ns GMT)"]]

    combined = (
        sent_frame
        # Join ack times.
        .join(acked_preprocessed, on=index, rsuffix=" acked")
        # Join experiment info.
        .join(exp_frame, on="expt_id")
        # Nicer labels for the two time columns.
        .rename(columns={
            "time (ns GMT)": "sent time (ns GMT)",
            "time (ns GMT) acked": "acked time (ns GMT)",
        })
        # A single session column instead of two, and transmit times.
        .assign(**{
            "session": lambda df: df["session_id"] + "_" + df["index"].astype(str),
            "trans_time": lambda df: (
                df["acked time (ns GMT)"] -
                df["sent time (ns GMT)"]
            ) / 1e9,  # ns -> s
        })
        .drop(columns=["session_id", "index", "expt_id"])
    )

    if 'cum_rebuf' not in combined:
        logging.debug(
            "No buffer data included for `%s`, loading manually", day)
        return download_and_match_buffer(day, combined, config)
    return combined


def download_and_match_buffer(day, frame, config: PufferExperimentConfig):
    """Load stalls and insert at matching timestamps in video_frame.

    As done in the puffer-statistics repo, we subtract the startup time from
    cum_rebuffer to get only stalls (rebuffering during play).
    """
    response = requests.get(config.get_puffer_data_url(day, "client_buffer"),
                            timeout=config.download_timeout)
    response.raise_for_status()
    buffer_csv = response.text

    started = {}
    times = defaultdict(list)
    rebuf = defaultdict(list)
    for event in _iter_csv(buffer_csv):
        key = f"{event['session_id']}_{event['index']}"
        event_time = int(event['time (ns GMT)'])
        eventtype = event['event']
        cum_rebuf = float(event['cum_rebuf'])
        if (eventtype == 'startup') and (key not in started):
            started[key] = cum_rebuf
        elif eventtype in ['startup', 'timer', 'play']:
            times[key].append(event_time)
            rebuf[key].append(cum_rebuf)

    # Only keep sessions with stalls and compute stall time.
    stalls = {
        key: np.array(rebuf[key]) - startup_delay
        for key, startup_delay in started.items()
        if any(item > startup_delay for item in rebuf[key])
    }

    # Not many session have stalls, so we match them in a simple loop
    # But first, ensure videos are sorted by timestamp
    frame = frame.sort_values(by='sent time (ns GMT)')
    frame['cum_rebuf'] = 0  # set default
    for session, session_stalls in stalls.items():
        # Sessions are tuples, we can't compare with ""=="" (pandas issue)
        index = frame['session'].isin([session])
        matched = (
            pd.Series(session_stalls, index=times[session])
            .sort_index()
            # match buffer to closest timestamp
            .asof(frame.loc[index, 'sent time (ns GMT)'])
            .fillna(method='ffill')  # Fill gaps with last valid value
            .fillna(0)  # Remaining gaps (e.g. in the beginning)
            .values  # drop index
        )
        frame.loc[index, 'cum_rebuf'] = matched

    return frame


def download_model_data(day, modeldir, config: PufferExperimentConfig,
                        model_version=1):
    """Download and extract model archive.

    As the archive needs to be extracted, a path must be specified.

    For some (rare) days, multiple models exist, and `model_version`
    can be used to choose between them.
    """
    model_url = config.get_puffer_data_url(day, "model", model_version)
    res = requests.get(model_url, timeout=config.download_timeout)

    with TemporaryDirectory() as tempdir:
        with tarfile.open(fileobj=BytesIO(res.content), mode='r:gz') as tar:
            tar.extractall(path=tempdir)

        # All files are extracted to a sub directory, copy content only
        subdirs = [
            subpath for subpath in Path(tempdir).iterdir()
            if subpath.is_dir()
        ]
        assert len(subdirs) == 1
        # Copy all files.
        for file in subdirs[0].iterdir():
            with open(file, 'rb') as src, \
                    open(modeldir / file.name, 'wb') as dst:
                dst.write(src.read())


# Utility functions.
# ==================

def load_inout(day, index=None, config=None) -> Union[InOut, InOutDict]:
    """Load inout data for a day; if provided, only for a specific model."""
    config = PufferExperimentConfig.with_updates(config)
    day = pd.to_datetime(day)
    daydir = config.get_data_directory(day)
    inout_files = [daydir / filename for filename in config.inout_filenames]
    frame_file = daydir / config.video_filename

    if index is None:
        inout_exists = any([file.is_file() for file in inout_files])
    else:
        inout_exists = inout_files[index].is_file()

    if inout_exists:
        if index is None:
            logging.debug("Loading all inout data from `%s`.", daydir)
            return [eh.data.read_npz(file) for file in inout_files]
        else:
            logging.debug("Loading inout data from `%s`.", inout_files[index])
            return eh.data.read_npz(inout_files[index])
    logging.debug("Computing inout data from `%s`.", frame_file)
    inout = to_ttp_inout(pd.read_csv(frame_file))
    return inout if index is None else inout[index]


def _iter_csv(csv_str: str):
    """Return iterator that returns rows from csv_str as dict."""
    datapoints = csv_str.splitlines()
    dialect = csv.Sniffer().sniff(datapoints[0])
    return csv.DictReader(datapoints, dialect=dialect)


def to_ttp_inout(frame: pd.DataFrame) -> InOut:
    """Scale and format a dataframe into inout data as used by puffer."""
    from .src_puffer import ttp  # pylint: disable=import-outside-toplevel
    sessions = _group_sessions(_scaled(frame))
    inout = ttp.prepare_input_output(sessions)
    # Optimize by casting to numpy arrays (particularly speed up storing).
    for model_data in inout:
        for key, values in model_data.items():
            model_data[key] = np.array(values)
    inout = cast(InOut, inout)  # Typing has problems with mutable type.
    return inout


def _scaled(frame: pd.DataFrame) -> pd.DataFrame:
    from .src_puffer import ttp  # pylint: disable=import-outside-toplevel
    scaling_factors = {
        'size': 1 / ttp.PKT_BYTES,
        'delivery_rate': 1 / ttp.PKT_BYTES,
        'cwnd': 1.0,
        'in_flight': 1.0,
        'min_rtt': 1 / ttp.MILLION,
        'rtt': 1 / ttp.MILLION,
        'ssim_index': 1.0,
        'trans_time': 1.0,
    }
    scaled = frame.copy()
    scaled[list(scaling_factors)] *= pd.Series(scaling_factors)
    return scaled


def _group_sessions(frame: pd.DataFrame):
    """Turn dataframe into a dict compatible with the ttp parser.

    Shape:
    {session -> video_ts -> row}
    """
    nested_dict: Dict[str, Dict[int, Dict]] = {}
    for _dict in frame.dropna(subset=["trans_time"]).to_dict(orient="records"):
        session = _dict.pop('session')
        video_ts = _dict.pop('video_ts')
        nested_dict.setdefault(session, {})[video_ts] = _dict
    return nested_dict


def puffer_sample_sizes(total_samples=1000000, discount=0.9, days=14):
    """Compute sample sizes from last to first day (most to least samples)."""
    weights = discount ** np.arange(days)  # Individual factor per day.
    return (total_samples * weights / np.sum(weights)).astype(int)
