"""PSI (Population Stability Index) constraint utilities.

PSI o day dinh nghia bang Jeffrey divergence, dung dung quy uoc cua
``optbinning.scorecard.monitoring`` (PSI = jeffrey(p_actual, p_expected)):

    PSI = sum_bins (p_fit - p_valid) * ln(p_fit / p_valid)

Moi bin cuoi la mot dai pre-bin lien tiep [j..i], nen dong gop PSI cua no la
mot hang so tinh truoc duoc. Nho vay tong PSI tren cac bin duoc chon co the
tuyen tinh hoa CHINH XAC bang dung thu thuat telescoping ma tac gia dung cho
ham muc tieu trong ``optbinning.binning.cp`` (dong 80-82):

    sum_i [ V[i][i]*x[i,i] + sum_{j<i} (V[i][j]-V[i][j+1])*x[i,j] ]

o day thay V bang ma tran PSI.
"""

import numpy as np

DEFAULT_M = int(1e6)
DEFAULT_EPS = 1e-6


def _flatten_special_codes(special_codes):
    """Trich cac special code dang so tu list/set/dict/scalar."""
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
            # bo qua special code khong phai so (numerical binning)
            pass
    return np.asarray(out, dtype=float)


def compute_prebin_counts(x_valid, splits, special_codes=None):
    """Dem so record cua tap valid roi vao tung pre-bin theo dung ``splits``.

    Tra ve mang int64 do dai ``len(splits) + 1``, khop voi mang
    ``n_nonevent``/``n_event`` (binary) hoac ``n_records`` (continuous) ma
    solver nhan. Loai bo missing (NaN) va special codes de khop voi tap
    "clean" ma pre-binning dung.
    """
    xv = np.asarray(x_valid, dtype=float).ravel()

    mask = ~np.isnan(xv)
    specials = _flatten_special_codes(special_codes)
    if specials.size:
        mask &= ~np.isin(xv, specials)
    xv = xv[mask]

    if xv.size == 0:
        raise ValueError("x_valid khong con record hop le sau khi loai "
                         "missing/special; khong the tinh PSI.")

    splits = np.asarray(splits, dtype=float)
    n_bins = len(splits) + 1
    indices = np.digitize(xv, splits, right=False)
    counts = np.bincount(indices, minlength=n_bins).astype(np.int64)
    return counts


def _psi_segment_matrix(n_fit, n_valid, M=DEFAULT_M, eps=DEFAULT_EPS):
    """Ma tran PSI[(i, j)] = round(dong gop PSI cua bin [j..i] * M), j <= i."""
    n_fit = np.asarray(n_fit, dtype=np.float64)
    n_valid = np.asarray(n_valid, dtype=np.float64)
    n = len(n_fit)

    if len(n_valid) != n:
        raise ValueError("n_fit va n_valid phai cung do dai; got {} vs {}."
                         .format(n, len(n_valid)))

    total_fit = n_fit.sum()
    total_valid = n_valid.sum()
    if total_fit <= 0 or total_valid <= 0:
        raise ValueError("Tong record fit/valid phai duong de tinh PSI.")

    # prefix sums de tong dai [j..i] tinh trong O(1)
    cum_fit = np.concatenate([[0.0], np.cumsum(n_fit)])
    cum_valid = np.concatenate([[0.0], np.cumsum(n_valid)])

    psi = {}
    for i in range(n):
        for j in range(i + 1):
            a = (cum_fit[i + 1] - cum_fit[j]) / total_fit
            e = (cum_valid[i + 1] - cum_valid[j]) / total_valid
            a = a if a > eps else eps
            e = e if e > eps else eps
            psi[(i, j)] = int(round((a - e) * np.log(a / e) * M))
    return psi


def add_psi_constraint_cp(model, n, x, n_fit, n_valid, psi_threshold,
                          M=DEFAULT_M, eps=DEFAULT_EPS):
    """Them rang buoc ``PSI(fit, valid) <= psi_threshold`` vao model CP-SAT.

    Parameters
    ----------
    model : ortools.sat.python.cp_model.CpModel
        Model da duoc ``BinningCP.build_model`` khoi tao (``self._model``).
    n : int
        So pre-bin (``self._n``).
    x : dict
        Bien quyet dinh ``x[i, j]`` (``self._x``).
    n_fit, n_valid : array-like, shape = (n,)
        So record moi pre-bin cua tap fit va tap valid.
    psi_threshold : float
        Nguong PSI toi da.
    """
    psi = _psi_segment_matrix(n_fit, n_valid, M=M, eps=eps)

    total_psi = sum(
        psi[(i, i)] * x[i, i] +
        sum((psi[(i, j)] - psi[(i, j + 1)]) * x[i, j] for j in range(i))
        for i in range(n)
    )

    model.Add(total_psi <= int(round(psi_threshold * M)))
