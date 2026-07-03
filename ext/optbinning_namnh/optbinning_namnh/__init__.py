"""optbinning_namnh: phan mo rong optbinning voi rang buoc PSI.

Cung cap cac estimator ke thua tu optbinning, them rang buoc PSI (Population
Stability Index) giua tap fit va tap valid vao bai toan toi uu binning. Neu
khong truyen psi_threshold/x_valid thi hanh vi giong het optbinning goc.
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
