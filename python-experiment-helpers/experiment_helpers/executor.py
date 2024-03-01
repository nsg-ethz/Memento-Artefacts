"""Function executors for the experiment framework.

Concretely, all executors expect a function and a list of pickled arguments,
and return a result equivalent to map(function, pickled_arguments).

We assume that the arguments have been pickled already and are save to be
passed to subprocesses etc.

When using SLURM, the function should return a value that can be pickled by
default pickle. Complex return objects are currently not supported, as they
are no needed by the framework. Here, we only return True or False.
"""
# pylint: disable=redefined-builtin

import logging
import pickle
import random
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import BoundedSemaphore, get_context
from textwrap import dedent

import cloudpickle

from .config import BaseConfig
from .data import Path


def map_experiments(func, iterable, *, config=None, logger=None):
    """Generic map function.

    Depending on config, either runs locally or using SLURM.
    """
    if logger is None:
        logger = logging.getLogger()
    config = BaseConfig.with_updates(config)

    if config.slurm:
        return map_slurm(func, iterable, config, logger)
    return map_local(func, iterable, config, logger)


def map_local(func, iterable, config: BaseConfig, logger: logging.Logger):
    """Run jobs locally.

    If n_jobs is 1, jobs are run sequentially, otherwise in parallel.
    """
    mp_context = get_context(config.multiprocessing_context)
    args = list(iterable)

    max_jobs = max(min(config.multiprocessing_jobs, len(args)), 1)
    if max_jobs == 1:
        logger.debug(
            "Starting `%i` job(s) (sequential).", len(args))
        return map(func, iterable)
    else:
        logger.debug("Starting `%i` job(s) (`%i` in parallel).",
                     len(args), max_jobs)
        with _BoundedPool(max_jobs, mp_context=mp_context) as pool:
            return pool.map(_func_pickled,
                            _pickle(func, args, config.random_start_delay))


def map_slurm(func, iterable, config: BaseConfig, logger: logging.Logger):
    """Run jobs with SLURM.

    Save function and arguments to files, and start a SLURM batch job.
    """
    jobs = _pickle(func, iterable, config.random_start_delay)
    maxjobs = config.slurm_max_array_size
    if (maxjobs is not None) and (len(jobs) > maxjobs):
        logger.debug("The number of jobs (`%i`) exceeds the maximum SLURM "
                     "array size (`%i`). Splitting into multiple rounds.",
                     len(jobs), maxjobs)
        results = []
        for i in range(0, len(jobs), maxjobs):
            results.extend(_map_slurm(jobs[i:i + maxjobs], config, logger))
        return results
    return _map_slurm(jobs, config, logger)


def _map_slurm(jobs, config: BaseConfig, logger: logging.Logger):
    """Run jobs with SLURM."""
    # Create a temporary directory for the run, inside the output directory,
    # as this must be read/writeable by the SLURM nodes.
    logger.debug(
        "Starting `%i` job(s) (using SLURM).", len(jobs))
    slurm_dir = Path(config.output_directory).absolute() / \
        f".slurm.{uuid.uuid4().hex}/"
    script_path = slurm_dir / "script.py"
    out_path = slurm_dir / "output.txt"   # Capture SLURM stdout and stderr.
    data_path = slurm_dir / "data.pickle"
    return_path = slurm_dir / "return.pickle"
    executor_cwd = Path.cwd().absolute()

    if config.slurm_log_dir is not None:
        logdir = Path(config.slurm_log_dir).absolute()
        logdir.mkdir(parents=True, exist_ok=True)
        logout = f"{logdir}/%j.out"
        logerr = f"{logdir}/%j.err"
    else:
        logout = logerr = "/dev/null"

    if config.slurm_simultaneous_jobs is not None:
        job_limit = f"%{config.slurm_simultaneous_jobs}"
    else:
        job_limit = ""

    if config.slurm_time_limit is not None:
        time_limit = f"#SBATCH --time={config.slurm_time_limit}"
    else:
        time_limit = ""

    # Create the SLURM script that calls the python script (see below).
    # Annotate the job with the command line arguments the CLI was started with
    # so we can identify the job, e.g. when using squeue.
    cmdline = shlex.join(sys.argv[1:])
    slurm_input = dedent(f"""\
        #!{config.slurm_shell}

        # SLURM options.
        #SBATCH --array=0-{len(jobs) - 1}{job_limit}
        #SBATCH --job-name="{cmdline}"
        #SBATCH --cpus-per-task={config.slurm_cpus}
        #SBATCH --gres=gpu:{config.slurm_gpus}
        #SBATCH --mem={config.slurm_mem}G
        #SBATCH --constraint="{config.slurm_constraints}"
        #SBATCH --output={logout}
        #SBATCH --error={logerr}
        #SBATCH --exclude="{config.slurm_exclude}"
        {time_limit}

        {config.slurm_interpreter} {script_path}
    """)
    logger.debug("---SLURM input---\n%s\n---input end---",
                 slurm_input.strip())

    # Define the python script that will be run by slurm.
    python_script = dedent(f"""\
        import os
        import pickle
        import sys

        # Start from same cwd as executor (for imports etc.)
        os.chdir("{executor_cwd}")
        sys.path.insert(0, "{executor_cwd}")

        # Get index fo current batch job.
        index = int(os.environ['SLURM_ARRAY_TASK_ID'])

        # Load function and corresponding to current job.
        with open(f'{data_path}.{{index}}', 'rb') as file:
            function, args = pickle.load(file)

        if __name__ == '__main__':
            # Execute.
            result = function(args)

            # Store return value.
            with open(f'{return_path}.{{index}}', 'wb') as file:
                pickle.dump(result, file)
    """)
    logger.debug("---SLURM script---\n%s\n---script end---",
                 python_script.strip())

    try:
        # Make dirs and write script and data.
        slurm_dir.mkdir(parents=True, exist_ok=True)
        with open(script_path, "w") as file:
            file.write(python_script)
        for index, data in enumerate(jobs):
            with open(f"{data_path}.{index}", "wb") as file:
                file.write(data)

        # Run with SLURM.
        # We don't check output as a whole; we want to know for each job
        # individually whether it failed or not by checking the output files.
        with open(out_path, "w+") as output_file:
            try:
                subprocess.run(
                    config.slurm_sbatch_cmd,
                    shell=True, input=slurm_input, text=True,
                    check=False, stdout=output_file, stderr=output_file,
                )
            except KeyboardInterrupt:
                output_file.seek(0)
                for line in output_file.read().splitlines():
                    if line.startswith("Submitted batch job "):
                        job_id = line.split()[-1]
                        break
                else:
                    # Interrupt before job was submitted.
                    job_id = None
                if job_id is not None:
                    logger.critical(
                        "KeyboardInterrupt received, "
                        f"cancelling SLURM job {job_id}. Please wait!")
                    subprocess.run(
                        f"{config.slurm_scancel_cmd} {job_id}",
                        shell=True, check=False,
                    )

        # Load return values.
        return_values = []
        failed = False
        for index in range(len(jobs)):
            try:
                with open(f"{return_path}.{index}", "rb") as file:
                    return_values.append(pickle.load(file))
            except FileNotFoundError:  # SLURM failed partially.
                return_values.append(None)
                failed = True

        if failed:
            # If the function fails, we can report the errors already.
            # If it is cancelled by SLURM, we point the user to the logs.
            logger.error(
                "Some SLURM job(s) failed or were cancelled. "
                "Enable and check the SLURM logs for more information."
            )
    finally:
        # Cleanup: remove the SLURM directory.
        shutil.rmtree(slurm_dir)

    return return_values


# Utilities.
# ==========

def _pickle(func, iterable, max_delay=None):
    """Pickle a function and a list of arguments."""
    args = list(iterable)
    if (len(args) > 1) and (max_delay is not None):
        delay = random.uniform(0, max_delay)
        func = partial(_delayed, func, delay)
    return [cloudpickle.dumps((func, arg)) for arg in args]


def _func_pickled(args):
    """Run a pickled function with pickled arguments."""
    try:
        function, argument = pickle.loads(args)
    except:  # pylint: disable=bare-except
        # This should not happen, but if it does, we want to know.
        logging.exception("Error during function setup!")
        return None
    return function(argument)


def _delayed(func, delay, args):
    time.sleep(delay)
    return func(args)


class _BoundedPool(ProcessPoolExecutor):
    """A pool that limits the number of concurrent workers.

    Otherwise we can run into Memory problems.

    Code adapted from:
    https://github.com/mowshon/bounded_pool_executor/blob/master/bounded_pool_executor/__init__.py
    """

    def __init__(self, max_workers: int = 1, **kwargs):
        self.semaphore = BoundedSemaphore(max_workers)
        super().__init__(max_workers=max_workers, **kwargs)

    def acquire(self):
        """Wait for resources."""
        self.semaphore.acquire()

    def release(self, fn):  # pylint: disable=invalid-name,unused-argument
        """Release resources."""
        self.semaphore.release()

    def submit(self, fn, /, *args, **kwargs):
        """Submit job, waiting for resources."""
        self.acquire()
        future = super().submit(fn, *args, **kwargs)
        future.add_done_callback(self.release)
        return future
