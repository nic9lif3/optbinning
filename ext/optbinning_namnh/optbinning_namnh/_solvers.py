"""Factory tao solver con: chi them rang buoc PSI sau ``super().build_model``.

Khong import truc tiep ``BinningCP``/``ContinuousBinningCP`` o day; ``base_cls``
duoc truyen tu ben ngoai (lay tu module goc ngay luc chay) de luon bam dung
class ma phien ban optbinning hien tai dang dung.
"""

import numpy as np

from ._psi import add_psi_constraint_cp


def make_psi_binning_cp(base_cls, n_valid, psi_threshold):
    """Subclass cua BinningCP (binary target) them rang buoc PSI.

    Fit counts moi pre-bin = n_nonevent + n_event.
    """

    class _PSIBinningCP(base_cls):
        def build_model(self, divergence, n_nonevent, n_event, trend_change):
            super().build_model(divergence, n_nonevent, n_event, trend_change)
            n_fit = np.asarray(n_nonevent) + np.asarray(n_event)
            add_psi_constraint_cp(self._model, self._n, self._x,
                                  n_fit, n_valid, psi_threshold)

    return _PSIBinningCP


def make_psi_continuous_binning_cp(base_cls, n_valid, psi_threshold):
    """Subclass cua ContinuousBinningCP (continuous target) them rang buoc PSI.

    Fit counts moi pre-bin = n_records.
    """

    class _PSIContinuousBinningCP(base_cls):
        def build_model(self, n_records, sums, ssums, trend_change):
            super().build_model(n_records, sums, ssums, trend_change)
            add_psi_constraint_cp(self._model, self._n, self._x,
                                  np.asarray(n_records), n_valid, psi_threshold)

    return _PSIContinuousBinningCP
