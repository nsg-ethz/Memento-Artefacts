#!/usr/bin/env python
"""A helper script to run all experiments over a uniform interface."""

import logging

import experiment_helpers as eh
from experiment_helpers import framework

# Import experiments so commands can register in CLI.
import experiments  # pylint: disable=unused-import

from experiments.puffer.implementation.test import test_training

# Also configure some debugging experiments to test the setup.


class DebugConfig(framework.BaseConfig):
    """Debug config to discard output and add some test attributes."""
    output_directory = eh.data.Path('/tmp/clidebug')
    debug_key = "Ensure that this shows up in output."


def echo():
    """Echo text for debugging, status, etc."""
    logging.critical("This is a critical log.")
    logging.error("This is an error log.")
    logging.warning("This is a warning log.")
    logging.info("This is an info log.")
    logging.debug("This is a debug log.")


def fail():
    """Debug command to test errors."""
    raise RuntimeError("Command failed succesfully.")


def show_config(config: DebugConfig):
    """Debug command to show config."""
    lines = []
    for attribute in dir(config):
        if attribute.startswith('__') or attribute == 'with_updates':
            continue
        lines.append(f"{attribute}: {getattr(config, attribute)}")
    joined = '\n'.join(lines)
    message = f"User config content:\n```\n{joined}\n```"
    logging.info(message)


# Idea: add another test for the remote?
run_debug = framework.ParametrizedExperiments("debug", {
    'echo': echo,
    'fail': fail,
    'show_config': show_config,
    'puffer-training': test_training,
}, DebugConfig)

# Start the CLI

if __name__ == "__main__":
    framework.experiment_cli()  # pylint: disable=no-value-for-parameter
