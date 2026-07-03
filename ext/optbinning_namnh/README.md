# optbinning_namnh

An extension of [optbinning](https://github.com/guillermo-navas-palencia/optbinning):
it adds a **PSI constraint** (Population Stability Index) between the fit set and
a validation set to the binning optimization problem.

The constraint: when the solver picks the optimal bins (maximizing IV), the
**PSI of the bins between the fit set and `x_valid` must be ≤ `psi_threshold`**.
PSI here is the Jeffrey divergence, following the convention of
`optbinning.scorecard.monitoring`.

## Installation

```bash
pip install ./ext/optbinning_namnh
# or build a wheel:
pip install build && python -m build ./ext/optbinning_namnh
```

## Usage

```python
from optbinning_namnh import PSIOptimalBinning, PSIContinuousOptimalBinning

# Binary target — same as OptimalBinning, plus psi_threshold + x_valid
ob = PSIOptimalBinning(solver="cp", psi_threshold=0.05)
ob.fit(x_train, y_train, x_valid=x_oot)
ob.binning_table.build()

# Without psi_threshold => identical behaviour to the base OptimalBinning
ob0 = PSIOptimalBinning(solver="cp")
ob0.fit(x_train, y_train)

# Continuous target
cb = PSIContinuousOptimalBinning(psi_threshold=0.05)
cb.fit(x_train, y_train, x_valid=x_oot)
```

Rule: `psi_threshold` and `x_valid` must be **both None** or **both provided**.

## How it works (no changes to the original source)

- **Exact PSI linearization**: each final bin is a contiguous run of pre-bins
  `[j..i]`, so its PSI contribution is a precomputable constant. The total PSI is
  expressed via the same telescoping trick optbinning uses for its objective
  (`optbinning/binning/cp.py`, lines 80-82), yielding a **linear** constraint in
  the CP-SAT model.
- **Solver layer**: subclass `BinningCP`/`ContinuousBinningCP`, overriding
  `build_model` = `super().build_model(...)` + adding the PSI constraint.
- **Estimator layer**: subclass `OptimalBinning`/`ContinuousOptimalBinning`. To
  **avoid copying** the ~145 lines of `_fit_optimizer`, we temporarily rebind the
  module-global name `BinningCP` to the PSI subclass and then call
  `super()._fit_optimizer(...)`.

## Limitations / notes (v1)

- Supports `dtype="numerical"` and `solver="cp"` only (continuous is always `cp`).
- **Pin the optbinning version** (`>=0.21,<0.22`): the package depends on internal
  APIs (`_fit_optimizer`, the `BinningCP` name, the `build_model` signature). When
  upgrading optbinning, re-run the tests.
- **Not thread-safe**: the module-global name rebinding is unsafe when running
  multiple fits in parallel within the same process (e.g. inside a multithreaded
  `BinningProcess`). Use it for independent per-variable binning.
- **`clone()`**: `psi_threshold` is not preserved by `sklearn.clone()` (it is
  outside `get_params`); set it again after cloning if used in a
  Pipeline/GridSearch.
- **Infeasibility**: PSI is a hard constraint. If `psi_threshold` is too tight the
  solver may fall back to a single bin (non-OPTIMAL status). Loosen
  `psi_threshold` in that case.

## Tests

```bash
pip install -e ".[test]"
pytest ext/optbinning_namnh/tests -v
```
