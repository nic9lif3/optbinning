# optbinning_namnh

Phần mở rộng của [optbinning](https://github.com/guillermo-navas-palencia/optbinning):
thêm **ràng buộc PSI** (Population Stability Index) giữa tập fit và tập valid vào
bài toán tối ưu binning.

Ràng buộc: khi solver chọn các bin tối ưu (maximize IV), **PSI của các bin giữa
tập fit và tập `x_valid` phải ≤ `psi_threshold`**. PSI ở đây = Jeffrey divergence,
đúng quy ước của `optbinning.scorecard.monitoring`.

## Cài đặt

```bash
pip install ./ext/optbinning_namnh
# hoặc build wheel:
pip install build && python -m build ./ext/optbinning_namnh
```

## Dùng

```python
from optbinning_namnh import PSIOptimalBinning, PSIContinuousOptimalBinning

# Binary target — y hệt OptimalBinning, thêm psi_threshold + x_valid
ob = PSIOptimalBinning(solver="cp", psi_threshold=0.05)
ob.fit(x_train, y_train, x_valid=x_oot)
ob.binning_table.build()

# Không truyền psi_threshold => hành vi giống hệt OptimalBinning gốc
ob0 = PSIOptimalBinning(solver="cp")
ob0.fit(x_train, y_train)

# Continuous target
cb = PSIContinuousOptimalBinning(psi_threshold=0.05)
cb.fit(x_train, y_train, x_valid=x_oot)
```

Quy tắc: `psi_threshold` và `x_valid` phải **cùng None** hoặc **cùng được cung cấp**.

## Cơ chế (không sửa source gốc)

- **PSI tuyến tính hóa chính xác**: mỗi bin cuối là một dải pre-bin liên tiếp
  `[j..i]` nên đóng góp PSI của nó là hằng số tính trước. Tổng PSI được biểu diễn
  bằng đúng thủ thuật telescoping mà optbinning dùng cho hàm mục tiêu
  (`optbinning/binning/cp.py` dòng 80-82), thành một ràng buộc **tuyến tính** trong
  mô hình CP-SAT.
- **Solver layer**: subclass `BinningCP`/`ContinuousBinningCP`, override
  `build_model` = `super().build_model(...)` + thêm ràng buộc PSI.
- **Estimator layer**: subclass `OptimalBinning`/`ContinuousOptimalBinning`. Để
  **không copy** 145 dòng `_fit_optimizer`, ta tạm hoán tên `BinningCP` trong
  module gốc bằng subclass PSI rồi gọi lại `super()._fit_optimizer(...)`.

## Giới hạn / Lưu ý (v1)

- Chỉ hỗ trợ `dtype="numerical"` và `solver="cp"` (continuous vốn luôn `cp`).
- **Ghim phiên bản optbinning** (`>=0.21,<0.22`): package phụ thuộc vào API nội
  bộ (`_fit_optimizer`, tên `BinningCP`, chữ ký `build_model`). Khi nâng optbinning,
  chạy lại test.
- **Không thread-safe**: cơ chế hoán tên module toàn cục không an toàn khi chạy
  nhiều fit song song trong cùng tiến trình (vd trong `BinningProcess` đa luồng).
  Dùng cho binning từng biến độc lập.
- **`clone()`**: `psi_threshold` không được `sklearn.clone()` giữ lại (nằm ngoài
  `get_params`); đặt lại sau khi clone nếu dùng trong Pipeline/GridSearch.
- **Vô nghiệm**: PSI là ràng buộc cứng. Nếu `psi_threshold` quá chặt, solver có
  thể trả về fallback 1 bin (status không OPTIMAL). Nới `psi_threshold` khi đó.

## Test

```bash
pip install -e ".[test]"
pytest ext/optbinning_namnh/tests -v
```
