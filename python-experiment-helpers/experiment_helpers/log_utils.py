"""Logging utilities.

1. a slackhandler, allowing logging messages to slack.
2. a log server, allowing to collect logs from spawned workers.
   (even if spawned by SLURM)

Solutions inspired by the Logging Cookbook:
https://docs.python.org/3/howto/logging-cookbook.html
"""
# pylint: disable=too-many-instance-attributes

import contextlib
import logging
import logging.handlers
import pickle
import queue
import sys
import time
import uuid

import click
import click_log
import fsspec
import requests  # type: ignore
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from .config import BaseConfig
from .data import Path
from .typing import Sequence, Tuple

# Client log contextmanager.
# ==========================


@contextlib.contextmanager
def client_logging(label: str, config: BaseConfig):
    """Client-side logging setup.

    We create files to send log records to the central server, i.e. the main
    framework process. This is the only method that works well with SLURM.

    We considered using file-based queues and unix sockets, but both did not
    cope well with SLURM, where we only have a shared file system.

    We use the root logger such that functions can use the default logging
    functions without caring for a specific logger.
    """
    root = logging.getLogger()
    original_level = root.level

    # A filter that adds experiment metadata.
    exp_filter = ExperimentNameFilter(label)

    # Can't use paths with dots or slashes for filenames.
    safe_label = file_safe_label(label)

    # Create a handler pickling records and a queue to not block while writing.
    client_queue = queue.Queue(-1)  # type: ignore
    queue_handler = logging.handlers.QueueHandler(client_queue)
    queue_handler.setLevel(config.base_loglevel)
    pickle_handler = PickleHandler(config.logpickledir, prefix=safe_label)
    pickle_handler.setLevel(config.base_loglevel)
    listener = logging.handlers.QueueListener(client_queue, pickle_handler)

    # Start everything. Stop and undo changes afterwards.
    try:
        listener.start()
        root.setLevel(config.base_loglevel)
        root.addFilter(exp_filter)
        root.addHandler(queue_handler)
        yield
    finally:
        listener.stop()
        # Cleanup.
        root.setLevel(original_level)
        root.removeHandler(queue_handler)
        root.removeFilter(exp_filter)
        # Create a file to indicate that we are done (no more new files).
        (Path(config.logpickledir) / f"{safe_label}.done").touch()


# Server log contextmanager.
# ==========================

@contextlib.contextmanager
def server_logging(name: str, config: BaseConfig):
    """Contextmanager to run a socker-based log server in a separate thread.

    Depending on config, create handlers for:
    - CLI,
    - slack,
    - files.
    """
    runlogger = logging.getLogger(f"{config.logbasename}.{name}")
    original_level = runlogger.level
    original_logpickledir = config.logpickledir

    # Queue so that handlers like slack (using the network) do not block.
    server_queue = queue.Queue(-1)  # type: ignore
    queue_handler = logging.handlers.QueueHandler(server_queue)
    queue_handler.setLevel(config.base_loglevel)

    # Prepare handlers and attach them to the queue.
    handlers = []
    click_handler = click_log.ClickHandler(config.loglevel)
    click_handler.formatter = ClickFormatter()
    handlers.append(click_handler)
    handlers.extend(make_filehandlers(config))
    slackhandler = make_slackhandler(  # The title is for the first message.
        config,
        # Exact command line arguments for easier re-runnning.
        title=f"`{' '.join(sys.argv)}`",
    )
    if slackhandler:
        handlers.append(slackhandler)
    listener = logging.handlers.QueueListener(server_queue, *handlers)

    # Create a file system watcher to collect logs from clients/experiments.
    random_dir = Path(config.output_directory).absolute() / \
        f".{uuid.uuid4().hex}"
    assert not random_dir.exists()
    if not config.slurm:
        # Locally we get full events and wait until a file is closed.
        observer = Observer()
        watcher = PickleWatcher(runlogger, "closed")
    else:
        # Using SLURM, we need to poll and only see files as "created".
        observer = PollingObserver(config.logpollinterval)
        watcher = PickleWatcher(runlogger, "created")

    # If slack is enabled, send a starting message for context.
    if slackhandler:
        slackhandler.send_start()

    try:
        # Configure the root logger.
        runlogger.setLevel(config.base_loglevel)
        runlogger.addHandler(queue_handler)

        # Create logpickledir, update config, and start watching.
        config.logpickledir = random_dir
        random_dir.mkdir(parents=True, exist_ok=True)
        observer.schedule(watcher, str(random_dir))

        # Start the threads.
        observer.start()
        listener.start()
        yield runlogger
    finally:
        # Stop the threads, reset values, and clean up.
        listener.stop()
        observer.stop()
        runlogger.setLevel(original_level)
        runlogger.removeHandler(queue_handler)
        config.logpickledir = original_logpickledir
        # Clean up.
        filesystem, fs_path = fsspec.core.url_to_fs(str(random_dir))
        filesystem.rm(fs_path, recursive=True)

    # If slack is enabled, send a completion message for context.
    if slackhandler:
        slackhandler.send_completed()


# Logging utilities.
# ==================


def file_safe_label(label):
    """Make label safe for use in filenames."""
    return label.replace(".", "").replace("/", "_")


def wait_for_logs(config: BaseConfig, arglist: Sequence[Tuple]):
    """Ensure that all log files have been processed."""
    logdir = Path(config.logpickledir)
    max_tries = config.logpolltimeout // config.logpollinterval
    current = 0
    # First wait for .done files indicating there there won't be any new logs.
    # They are created if the function completes or crashes.
    done_files = [logdir / (file_safe_label(arg[0]) + '.done')
                  for arg in arglist]
    for file in done_files:
        while (not file.exists()) and current < max_tries:
            current += 1
            time.sleep(config.logpollinterval)

    # No new logfiles, ensure that all existing ones have been processed.
    while any(logdir.glob("*.log")) and current < max_tries:
        current += 1
        time.sleep(config.logpollinterval)


def make_slackhandler(config: BaseConfig, title: str):
    """Create a handler sending messages to thread."""
    if not (config.slacktoken and config.slackchannel):
        return None  # Not enough data.

    # Note that "thread" here is a thread on Slack, not a system thread.
    slackthread = SlackThread(
        config.slacktoken, config.slackchannel, title=title)
    slackthread.setLevel(config.loglevel)

    return slackthread


def make_filehandlers(config: BaseConfig):
    """Configure filehandlers, if any."""
    # For logfiles, we can use more extensive logging.
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s",
                                  datefmt=config.utcfmt)
    # Ensure UTC.
    formatter.converter = time.gmtime  # type: ignore

    handlers = []
    for loggername, filename in config.logfiles.items():
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(loggername)
        logger.setLevel(logging.DEBUG)
        handler = logging.handlers.RotatingFileHandler(
            filename, **config.logfile_kwargs
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        handlers.append(handler)

    return handlers


class ExperimentNameFilter(logging.Filter):
    """A logging filter to add the experiment name to records."""

    def __init__(self, experiment: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment = experiment

    def filter(self, record):
        """Add experiment name to record."""
        record.experiment = self.experiment
        record.experiment_prefix = f'`{self.experiment}` '
        record.msg = f'{record.experiment_prefix}{record.msg}'
        return True


class ClickFormatter(logging.Formatter):
    """Formatter using the info from ExperimentNameFilter.

    Similar to the click_log formatter, but use the experiment name.
    Also, save screen space by not writing log levels; just use the coloring.

    Source:
    https://github.com/click-contrib/click-log/blob/master/click_log/core.py
    """
    colors = {
        'error': dict(fg='red'),
        'exception': dict(fg='red'),
        'critical': dict(fg='red'),
        'debug': dict(fg='blue'),
        'warning': dict(fg='yellow'),
    }

    def format(self, record):
        terminal_prefix = ""
        msg = record.getMessage()
        if hasattr(record, 'experiment'):
            # Remove prefix from message and add it for each line as that looks
            # nicer and more uniform in the terminal
            terminal_prefix = f'{record.experiment}: '
            msg = msg.lstrip(record.experiment_prefix)

        style = self.colors.get(record.levelname.lower(), {})
        return '\n'.join(click.style(terminal_prefix + x, **style)
                         for x in msg.splitlines())


class PickleHandler(logging.Handler):
    """A handler pickling records and saving them to a random file."""

    def __init__(self, directory, *args, prefix="", **kwargs):
        super().__init__(*args, **kwargs)
        self.directory = Path(directory)
        self.prefix = prefix

    def emit(self, record):
        """Pickle record."""
        # When using pdb to debug experiment function, we run into strange
        # issues. pdb seems to somehow modify logging in the background
        # and we observe two inifinite event loops, that we prevent as follows:
        # 1: Do not pickle records that come from watchdog file events.
        if record.name.startswith("watchdog."):
            return
        # 2: Pickle each record only once.
        if hasattr(record, "pickled"):
            return
        record.pickled = True

        # Create a random file.
        while True:
            random_file = self.directory / \
                f"{self.prefix}.{uuid.uuid4().hex}.log"
            if not random_file.exists():
                break
        # Pickle record.
        with random_file.open("wb") as file:
            pickle.dump(record, file)


class PickleWatcher(FileSystemEventHandler):
    """A handler watching a directory for new files and processing them.

    Events that were sent to the root logger go to `self.logger`.
    Events that were sent to specific logger go to the original logger.
    """

    def __init__(self, logger, event, *args, attempts=10, sleep=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.event = event
        self.attempts = attempts
        self.sleep = sleep

    def on_any_event(self, event):
        """Process event if it has the correct type."""
        path = event.src_path
        if path.endswith(".log") and (event.event_type == self.event):
            try:
                for attempt in range(1, self.attempts + 1):
                    try:
                        with open(path, "rb") as file:
                            record = pickle.load(file)
                        break  # Success!
                    except:  # pylint: disable=bare-except
                        if attempt < self.attempts:
                            time.sleep(self.sleep)
                            continue
                        self.logger.warning("Could not process a log record.")
                        return
            finally:
                # Remove file whether it could be parsed or not.
                Path(path).unlink()

            if record.name == "root":
                self.logger.handle(record)
            else:
                logging.getLogger(record.name).handle(record)


# Slack logging.
# ==============

class SlackThread(logging.Handler):
    """Send messages to a slack thread.

    May be used as a logging handler.

    The first message starts a new thread, all following messages are sent
    as replies.

    In debugging mode, http errors are raised.
    Otherwise, they are silently ignored, such that issues with slack to not
    crash the main program.

    Create your app here: https://api.slack.com/apps/
    Find the channel URL by right clicking on your channel, selecting
    "Copy link", which will result in a link like
    `https://workspace.slack.com/archives/XYZ`, where `XYZ` is the channel id.
    """
    POST_URL = "https://slack.com/api/chat.postMessage"
    UPDATE_URL = "https://slack.com/api/chat.update"
    LINK_URL = "https://slack.com/api/chat.getPermalink"

    def __init__(self, token, channel, title=None, thread=None,
                 debug=False, broadcast_problems=False, timeout=1):
        super().__init__()
        self.token = token
        self.channel = channel
        self.debug = debug
        self.broadcast_problems = broadcast_problems
        self.timeout = timeout

        # Formatting of root message.
        self.title = title  # f"`{title}`" if title else ""
        self.status_icon = ""

        # Thread data!
        self.thread = thread
        self.root_ts = None
        self.root_url = None

    def send(self, message, broadcast=False, update=False):
        """Send a message.

        Starts a new thread, if no thread exsits, replies otherwise.
        If broadcast=True, the reply is sent to the channel as well.

        If update=True, edit root message of thread, and append message
        to the original message (separated by newline).
        """
        headers = {
            'Authorization': f"Bearer {self.token}"
        }
        data = {"text": message, "channel": self.channel,
                "unfurl_links": False}
        if self.thread is not None:
            data['thread_ts'] = self.thread
            if broadcast:
                data["reply_broadcast"] = True
        try:
            response = requests.post(
                self.POST_URL, headers=headers, json=data, timeout=self.timeout
            )
            response.raise_for_status()
            if not response.json()["ok"]:
                raise RuntimeError(response.text)
            if self.thread is None:
                res = response.json()
                self.thread = res['ts']
                self.root_ts = res["message"]['ts']
        except (requests.ConnectionError, requests.HTTPError, RuntimeError) \
                as error:
            if self.debug:  # Only raise in debug mode.
                raise error

        if update and self.thread is not None and self.root_ts is not None:
            self.update_message(
                self.root_ts, f"{self.status_icon} {update} {self.title}")

    def update_message(self, message_ts, new_message):
        """Update a message."""
        headers = {
            'Authorization': f"Bearer {self.token}"
        }
        data = {"token": self.token,
                "channel": self.channel,
                "ts": message_ts,
                "text": new_message}
        try:
            response = requests.post(self.UPDATE_URL,
                                     headers=headers, json=data,
                                     timeout=self.timeout)
            response.raise_for_status()
            if not response.json()["ok"]:
                raise RuntimeError(response.text)
        except (requests.ConnectionError, requests.HTTPError, RuntimeError) \
                as error:
            if self.debug:  # Only raise in debug mode.
                raise error

    def get_root_url(self):
        """Get permalink to root message, e.g. to link from other threads."""
        headers = {'Authorization': f"Bearer {self.token}"}
        data = {"message_ts": self.root_ts, "channel": self.channel}
        try:
            response = requests.get(
                self.LINK_URL, headers=headers, params=data,
                timeout=self.timeout)
            if not response.json()["ok"]:
                raise RuntimeError(response.text)
            return response.json()["permalink"]
        except (requests.ConnectionError, requests.HTTPError, RuntimeError) \
                as error:
            if self.debug:  # Only raise in debug mode.
                raise error
            return None

    def send_start(self):
        """Send a "experiment starting" message."""
        self.send(f":gear: *Running:* {self.title}")

    def send_completed(self):
        """Send a completed message.

        If warnings or erros occured, show that icon instead of a checkmark.
        """
        if not self.status_icon:  # Do not override errors and warnings.
            self.status_icon = ":heavy_check_mark:"
        self.send("Done.", update="*Completed:*")

    def send_error(self, error=""):
        """Send an error message, broadcast to channel."""
        # Split traceback
        if "Traceback" in error:
            message, traceback = error.split("Traceback", maxsplit=1)
            lines = traceback.splitlines()
            if len(lines) > 1:
                traceback = f"(...)\n{lines[-1]}"
            # traceback = "Traceback" + traceback
        else:
            message = error
            traceback = ""

        self.status_icon = ":x:"  # Override.
        message = f":x: {message}"
        if traceback:
            message += f"\n```{traceback}```"
        self.send(message, update="*Failed:*",
                  broadcast=self.broadcast_problems)

    def send_warning(self, warning):
        """Send a warning."""
        self.status_icon = ":warning:"
        self.send(":warning: " + warning,
                  update="*Running:*",
                  broadcast=self.broadcast_problems)

    def emit(self, record: logging.LogRecord):
        """Method that is called as a logging handler."""
        msg = self.format(record)
        if record.levelno in (logging.ERROR, logging.CRITICAL):
            self.send_error(msg)
        elif record.levelno == logging.WARNING:
            self.send_warning(msg)
        else:
            self.send(msg)
