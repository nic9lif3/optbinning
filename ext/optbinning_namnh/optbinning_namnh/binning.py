"""PSIOptimalBinning = OptimalBinning + rang buoc PSI giua tap fit va tap valid.

Ke thua toan bo ``optbinning.OptimalBinning`` va KHONG sua source goc. Thay doi:
  - ``__init__``: them ``psi_threshold`` (mac dinh None).
  - ``fit``: them ``x_valid`` (mac dinh None) de kiem tra rang buoc PSI khi
    solver lua chon kich ban bin toi uu.

Neu ``psi_threshold is None`` -> hanh vi giong het ``OptimalBinning`` goc.

Ky thuat: khong copy ``_fit_optimizer`` (145 dong logic monotonic/bin-size cua
tac gia). Thay vao do, tam hoan doi ten ``BinningCP`` trong module goc bang mot
subclass co them rang buoc PSI, roi goi lai ``super()._fit_optimizer(...)``.
"""

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
        # Dung nguyen danh sach param cua lop cha:
        #   - Tu dong cap nhat neu upstream them param moi (forward-compat,
        #     khong phai liet ke lai ~35 tham so).
        #   - CO Y bo 'psi_threshold' de _fit goi _check_parameters(
        #     **self.get_params()) khong bi vo (ham do khong nhan param nay).
        # Danh doi: sklearn clone() se khong giu psi_threshold (dat lai sau
        # khi clone neu can dung trong Pipeline/GridSearch...).
        return OptimalBinning._get_param_names()

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

        if self.solver != "cp":
            raise ValueError(
                "Rang buoc PSI hien chi ho tro solver='cp'; got '{}'."
                .format(self.solver))

    def fit(self, x, y, sample_weight=None, check_input=False, x_valid=None):
        """Fit optimal binning co rang buoc PSI.

        Parameters
        ----------
        x, y : array-like, shape = (n_samples,)
            Vector huan luyen va target nhi phan.
        sample_weight : array-like, optional
        check_input : bool (default=False)
        x_valid : array-like, optional (default=None)
            Tap valid dung de tinh PSI. Bat buoc cung/khong-cung ton tai voi
            ``psi_threshold``.
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
        # Khong co PSI, hoac solver khong chay (<=1 pre-bin) -> y het lop cha.
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
