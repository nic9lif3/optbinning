"""optbinning_namnh: optbinning extension with a PSI constraint.

Provides estimators that inherit from optbinning and add a PSI (Population
Stability Index) constraint between the fit and validation sets to the binning
optimization problem. If psi_threshold/x_valid are not provided, the behaviour
is identical to the base optbinning estimators.
"""

from ._psi import add_psi_constraint_cp
from ._psi import compute_prebin_counts
from .binning import PSIOptimalBinning
from .continuous_binning import PSIContinuousOptimalBinning

__version__ = "0.1.0"

__all__ = [
    "PSIOptimalBinning",
    "PSIContinuousOptimalBinning",
    "add_psi_constraint_cp",
    "compute_prebin_counts",
    "__version__",
]
