"""PSIContinuousOptimalBinning = ContinuousOptimalBinning + rang buoc PSI.

Cung nguyen tac voi ``PSIOptimalBinning`` (xem binning.py). ContinuousOptimalBinning
luon dung solver 'cp' nen khong can kiem tra solver.
"""

import optbinning.binning.continuous_binning as _cont_mod
from optbinning import ContinuousOptimalBinning

from ._psi import compute_prebin_counts
from ._solvers import make_psi_continuous_binning_cp


class PSIContinuousOptimalBinning(ContinuousOptimalBinning):
    def __init__(self, psi_threshold=None, **kwargs):
        super().__init__(**kwargs)
        self.psi_threshold = psi_threshold
        self._x_valid = None

    @classmethod
    def _get_param_names(cls):
        # Xem giai thich trong PSIOptimalBinning._get_param_names.
        return ContinuousOptimalBinning._get_param_names()

    def _validate_psi_inputs(self, x_valid):
        if (self.psi_threshold is None) != (x_valid is None):
            raise ValueError(
                "psi_threshold va x_valid phai cung None hoac cung duoc cung "
                "cap. psi_threshold={}, x_valid is None: {}."
                .format(self.psi_threshold, x_valid is None))

        if self.psi_threshold is None:
            return

        if (not isinstance(self.psi_threshold, (int, float)) or
                isinstance(self.psi_threshold, bool) or
                self.psi_threshold <= 0):
            raise ValueError("psi_threshold phai la so duong; got {}."
                             .format(self.psi_threshold))

        if self.dtype != "numerical":
            raise ValueError(
                "Rang buoc PSI hien chi ho tro dtype='numerical'.")

    def fit(self, x, y, sample_weight=None, check_input=False, x_valid=None):
        """Fit continuous optimal binning co rang buoc PSI.

        Parameters
        ----------
        x, y : array-like, shape = (n_samples,)
            Vector huan luyen va target continuous.
        sample_weight : array-like, optional
        check_input : bool (default=False)
        x_valid : array-like, optional (default=None)
            Tap valid dung de tinh PSI. Bat buoc cung/khong-cung ton tai voi
            ``psi_threshold``.
        """
        self._validate_psi_inputs(x_valid)
        self._x_valid = x_valid
        return self._fit(x, y, sample_weight, check_input)

    def fit_transform(self, x, y, sample_weight=None, metric="mean",
                      metric_special=0, metric_missing=0, show_digits=2,
                      check_input=False, x_valid=None):
        return self.fit(x, y, sample_weight, check_input, x_valid).transform(
            x, metric, metric_special, metric_missing, show_digits,
            check_input)

    def _fit_optimizer(self, splits, n_records, sums, ssums, stds):
        if (self.psi_threshold is None or self._x_valid is None or
                len(n_records) <= 1):
            return super()._fit_optimizer(splits, n_records, sums, ssums, stds)

        n_valid = compute_prebin_counts(self._x_valid, splits,
                                        self.special_codes)

        factory = make_psi_continuous_binning_cp(
            _cont_mod.ContinuousBinningCP, n_valid, self.psi_threshold)
        original = _cont_mod.ContinuousBinningCP
        _cont_mod.ContinuousBinningCP = factory
        try:
            return super()._fit_optimizer(splits, n_records, sums, ssums, stds)
        finally:
            _cont_mod.ContinuousBinningCP = original
