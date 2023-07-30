"""Default imports for ease of use."""

from . import alternatives, bases, distances, metrics, models, utils
# Main models and distances for easier access..
from .distances import (JSDCategoricalDistribution, JSDCategoricalPoint,
                        JSDContinuousDistribution, JSDContinuousPoint)
from .models import MultiMemory, Memento
