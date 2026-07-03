"""PSI (Population Stability Index) constraint utilities.

PSI here is defined as the Jeffrey divergence, following the same convention as
``optbinning.scorecard.monitoring`` (PSI = jeffrey(p_actual, p_expected)):

    PSI = sum_bins (p_fit - p_valid) * ln(p_fit / p_valid)

Each final bin is a contiguous run of pre-bins [j..i], so its PSI contribution
is a precomputable constant. Therefore the total PSI over the selected bins can
be linearized EXACTLY using the same telescoping trick the author uses for the
objective in ``optbinning.binning.cp`` (lines 80-82):

    sum_i [ V[i][i]*x[i,i] + sum_{j<i} (V[i][j]-V[i][j+1])*x[i,j] ]

replacing V with a PSI matrix.
"""

import numpy as np

DEFAULT_M = int(1e6)
# Empty/near-empty bins: their proportion is floored at this value so PSI stays
# finite (avoids ln of 0). 1e-3 means an empty bin is treated as 0.1% mass.
DEFAULT_EPS = 1e-3


def _flatten_special_codes(special_codes):
    """Extract numeric special codes from a list/set/dict/scalar."""
    if special_codes is None:
        return np.array([], dtype=float)

    if isinstance(special_codes, dict):
        values = []
        for v in special_codes.values():
            if isinstance(v, (list, tuple, set, np.ndarray)):
                values.extend(list(v))
            else:
                values.append(v)
    elif isinstance(special_codes, (list, tuple, set, np.ndarray)):
        values = list(special_codes)
    else:
        values = [special_codes]

    out = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            # skip non-numeric special codes (numerical binning only)
            pass
    return np.asarray(out, dtype=float)


def compute_prebin_counts(x_valid, splits, special_codes=None):
    """Count validation-set records falling into each pre-bin, using ``splits``.

    Returns an int64 array of length ``len(splits) + 1``, aligned with the
    ``n_nonevent``/``n_event`` (binary) or ``n_records`` (continuous) arrays the
    solver receives. Missing (NaN) and special codes are removed so the counts
    match the "clean" set used by pre-binning.
    """
    xv = np.asarray(x_valid, dtype=float).ravel()

    mask = ~np.isnan(xv)
    specials = _flatten_special_codes(special_codes)
    if specials.size:
        mask &= ~np.isin(xv, specials)
    xv = xv[mask]

    if xv.size == 0:
        raise ValueError("x_valid has no valid records left after removing "
                         "missing/special values; cannot compute PSI.")

    splits = np.asarray(splits, dtype=float)
    n_bins = len(splits) + 1
    indices = np.digitize(xv, splits, right=False)
    counts = np.bincount(indices, minlength=n_bins).astype(np.int64)
    return counts


def _psi_segment_matrix(n_fit, n_valid, M=DEFAULT_M, eps=DEFAULT_EPS):
    """Matrix PSI[(i, j)] = round(PSI contribution of bin [j..i] * M), j <= i."""
    n_fit = np.asarray(n_fit, dtype=np.float64)
    n_valid = np.asarray(n_valid, dtype=np.float64)
    n = len(n_fit)

    if len(n_valid) != n:
        raise ValueError("n_fit and n_valid must have the same length; got {} "
                         "vs {}.".format(n, len(n_valid)))

    total_fit = n_fit.sum()
    total_valid = n_valid.sum()
    if total_fit <= 0 or total_valid <= 0:
        raise ValueError("Total fit/valid records must be positive to compute "
                         "PSI.")

    # prefix sums so a segment [j..i] total is computed in O(1)
    cum_fit = np.concatenate([[0.0], np.cumsum(n_fit)])
    cum_valid = np.concatenate([[0.0], np.cumsum(n_valid)])

    psi = {}
    for i in range(n):
        for j in range(i + 1):
            a = (cum_fit[i + 1] - cum_fit[j]) / total_fit
            e = (cum_valid[i + 1] - cum_valid[j]) / total_valid
            # floor empty/near-empty proportions so ln(a/e) stays finite
            a = a if a > eps else eps
            e = e if e > eps else eps
            psi[(i, j)] = int(round((a - e) * np.log(a / e) * M))
    return psi


def add_psi_constraint_cp(model, n, x, n_fit, n_valid, psi_threshold,
                          M=DEFAULT_M, eps=DEFAULT_EPS):
    """Add the constraint ``PSI(fit, valid) <= psi_threshold`` to a CP-SAT model.

    Parameters
    ----------
    model : ortools.sat.python.cp_model.CpModel
        Model already initialized by ``BinningCP.build_model`` (``self._model``).
    n : int
        Number of pre-bins (``self._n``).
    x : dict
        Decision variables ``x[i, j]`` (``self._x``).
    n_fit, n_valid : array-like, shape = (n,)
        Per-pre-bin record counts of the fit and validation sets.
    psi_threshold : float
        Maximum allowed PSI.
    """
    psi = _psi_segment_matrix(n_fit, n_valid, M=M, eps=eps)

    total_psi = sum(
        psi[(i, i)] * x[i, i] +
        sum((psi[(i, j)] - psi[(i, j + 1)]) * x[i, j] for j in range(i))
        for i in range(n)
    )

    model.Add(total_psi <= int(round(psi_threshold * M)))
