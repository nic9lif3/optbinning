"""PSIOptimalBinning = OptimalBinning + a PSI constraint between fit and valid.

Inherits the whole ``optbinning.OptimalBinning`` and does NOT modify its source.
Changes:
  - ``__init__``: adds ``psi_threshold`` (default None).
  - ``fit``: adds ``x_valid`` (default None) used to enforce the PSI constraint
    while the solver picks the optimal binning scenario.

If ``psi_threshold is None`` the behaviour is identical to the base
``OptimalBinning``.

Technique: we do NOT copy the ~145 lines of ``_fit_optimizer`` (the author's
monotonic/bin-size logic). Instead we temporarily rebind the module-global name
``BinningCP`` to a PSI subclass, then call ``super()._fit_optimizer(...)``.
"""

import numpy as np

import optbinning.binning.binning as _binning_mod
from optbinning import OptimalBinning

from ._psi import compute_prebin_counts
from ._solvers import make_psi_binning_cp


class PSIOptimalBinning(OptimalBinning):
    def __init__(self, psi_threshold=None, **kwargs):
        super().__init__(**kwargs)
        self.psi_threshold = psi_threshold
        self._x_valid = None

    @classmethod
    def _get_param_names(cls):
        # Reuse the parent's parameter list:
        #   - auto-updates if upstream adds a new parameter (forward-compat, no
        #     need to re-list the ~35 parameters).
        #   - deliberately excludes 'psi_threshold' so that _fit's call
        #     _check_parameters(**self.get_params()) does not break (that
        #     function does not accept this parameter).
        # Trade-off: sklearn clone() will not preserve psi_threshold (set it
        # again after cloning if used inside a Pipeline/GridSearch...).
        return OptimalBinning._get_param_names()

    def _validate_psi_inputs(self, x_valid):
        if (self.psi_threshold is None) != (x_valid is None):
            raise ValueError(
                "psi_threshold and x_valid must both be None or both be "
                "provided. psi_threshold={}, x_valid is None: {}."
                .format(self.psi_threshold, x_valid is None))

        if self.psi_threshold is None:
            return

        if (not isinstance(self.psi_threshold, (int, float)) or
                isinstance(self.psi_threshold, bool) or
                self.psi_threshold <= 0):
            raise ValueError("psi_threshold must be a positive number; got {}."
                             .format(self.psi_threshold))

        if self.dtype != "numerical":
            raise ValueError(
                "The PSI constraint currently supports dtype='numerical' only.")

        if self.solver != "cp":
            raise ValueError(
                "The PSI constraint currently supports solver='cp' only; got "
                "'{}'.".format(self.solver))

        xv = np.asarray(x_valid, dtype=float).ravel()
        if xv.size == 0 or np.all(np.isnan(xv)):
            raise ValueError(
                "x_valid is empty or all-missing; cannot compute PSI.")

    def fit(self, x, y, sample_weight=None, check_input=False, x_valid=None):
        """Fit optimal binning with a PSI constraint.

        Parameters
        ----------
        x, y : array-like, shape = (n_samples,)
            Training vector and binary target.
        sample_weight : array-like, optional
        check_input : bool (default=False)
        x_valid : array-like, optional (default=None)
            Validation set used to compute PSI. Must be provided/absent together
            with ``psi_threshold``.
        """
        self._validate_psi_inputs(x_valid)
        self._x_valid = x_valid
        return self._fit(x, y, sample_weight, check_input)

    def fit_transform(self, x, y, sample_weight=None, metric="woe",
                      metric_special=0, metric_missing=0, show_digits=2,
                      check_input=False, x_valid=None):
        return self.fit(x, y, sample_weight, check_input, x_valid).transform(
            x, metric, metric_special, metric_missing, show_digits,
            check_input)

    def _fit_optimizer(self, splits, n_nonevent, n_event):
        # No PSI, or the solver is not run (<=1 pre-bin) -> behave like base.
        if (self.psi_threshold is None or self._x_valid is None or
                len(n_nonevent) <= 1):
            return super()._fit_optimizer(splits, n_nonevent, n_event)

        n_valid = compute_prebin_counts(self._x_valid, splits,
                                        self.special_codes)

        factory = make_psi_binning_cp(_binning_mod.BinningCP, n_valid,
                                      self.psi_threshold)
        original = _binning_mod.BinningCP
        _binning_mod.BinningCP = factory
        try:
            return super()._fit_optimizer(splits, n_nonevent, n_event)
        finally:
            _binning_mod.BinningCP = original
