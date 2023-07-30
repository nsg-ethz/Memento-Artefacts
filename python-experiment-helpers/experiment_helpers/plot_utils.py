"""Plot import and configuration.

If I need a helper for logplots: https://stackoverflow.com/a/70758301
"""
# pylint: disable=invalid-name

import warnings
from functools import partial

import numpy as np
import seaborn as sns

# Seaborn sets the figure layout to tight, which raises a warning. Ignore.
warnings.filterwarnings("ignore",
                        message="The figure layout has changed to tight")


# Note: I found it easiest to work with plots
# At double the size in python, and use scale=0.5 in latex.

# For some reason, matplotlib gets finnicky with small font sizes,
# so just working with double size and scaling down worked better!

def setup(latex=True, rcparams=None):
    _rc = {
        'text.usetex': latex,
        # 'font.family': 'serif',
        'text.latex.preamble': r'\usepackage{amsmath,amssymb} \usepackage{mathptmx}',

        'axes.labelpad': 4,
        'xtick.major.pad': 0,
        'ytick.major.pad': 0,

        'lines.linewidth': 2,
        'lines.markeredgewidth': 1,
        'lines.markeredgecolor': 'black',
        'lines.markersize': 10,

        'scatter.marker': '.',
        'scatter.edgecolors': 'none',

        # Set image quality and reduce whitespace around saved figure.
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    }
    if rcparams:
        _rc.update(rcparams)
    # default scale for paper is 0.8, double that because we scale down.
    sns.set_theme("paper", "whitegrid", font_scale=1.6, rc=_rc)


default_width = 3.54 * 0.9  # 3.54 is the width of a column, but leave space.
default_height = 2 * default_width / np.sqrt(2)
default_aspect = 1 * np.sqrt(2)

default_size = (default_height * default_aspect, default_height)
default_size_short = (default_height * default_aspect, default_height / 1.5)

default_opts = dict(
    height=default_height,
    aspect=default_aspect,
    facet_kws=dict(despine=False, legend_out=False,),
)
default_opts_square = dict(
    height=default_height,
    aspect=1,
    facet_kws=dict(despine=False, legend_out=False,),
)
default_opts_out = dict(
    height=default_height,
    aspect=default_aspect,
    facet_kws=dict(despine=False, legend_out=True,),
)


onethird_ops = dict(  # Scale such that three plots fit next to each other better.
    height=2.5 / np.sqrt(2),
    aspect=1,
    facet_kws=dict(despine=False, legend_out=False,),
)


def opts(facet_kws=None, is_grid=False, **kws):
    _base = dict(
        height=default_height,
        aspect=default_aspect,
    )
    _facet_kws = dict(despine=False, legend_out=False,)
    if kws:
        _base.update(kws)
    if facet_kws:
        _facet_kws.update(facet_kws)
    if not is_grid:
        _base['facet_kws'] = _facet_kws
    else:
        _base.update(_facet_kws)

    return _base


opts_square = partial(opts, aspect=1)
opts_square_small = partial(opts, height=default_height*0.8, aspect=1)
# Scale for three plots in the top row.
opts_onethird = partial(
    opts,
    height=0.95 * 2 * 2.4 / np.sqrt(2),
    # aspect=1 / 0.8,
)

# Two-thirds of normal height.
opts_short = partial(
    opts,
    height=default_height / 1.5,
    aspect=default_aspect * 1.5,
)
