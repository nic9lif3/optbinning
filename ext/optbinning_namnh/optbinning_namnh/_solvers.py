"""Factories for solver subclasses that only add the PSI constraint after
``super().build_model``.

``BinningCP``/``ContinuousBinningCP`` are not imported directly here; ``base_cls``
is passed in from the outside (read from the original module at call time) so we
always bind to whatever class the current optbinning version actually uses.
"""

import numpy as np

from ._psi import add_psi_constraint_cp


def make_psi_binning_cp(base_cls, n_valid, psi_threshold):
    """Subclass of BinningCP (binary target) that adds the PSI constraint.

    Per-pre-bin fit counts = n_nonevent + n_event.
    """

    class _PSIBinningCP(base_cls):
        def build_model(self, divergence, n_nonevent, n_event, trend_change):
            super().build_model(divergence, n_nonevent, n_event, trend_change)
            n_fit = np.asarray(n_nonevent) + np.asarray(n_event)
            add_psi_constraint_cp(self._model, self._n, self._x,
                                  n_fit, n_valid, psi_threshold)

    return _PSIBinningCP


def make_psi_continuous_binning_cp(base_cls, n_valid, psi_threshold):
    """Subclass of ContinuousBinningCP (continuous target) that adds the PSI
    constraint.

    Per-pre-bin fit counts = n_records.
    """

    class _PSIContinuousBinningCP(base_cls):
        def build_model(self, n_records, sums, ssums, trend_change):
            super().build_model(n_records, sums, ssums, trend_change)
            add_psi_constraint_cp(self._model, self._n, self._x,
                                  np.asarray(n_records), n_valid, psi_threshold)

    return _PSIContinuousBinningCP
