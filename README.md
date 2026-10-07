# ks_convergence_analysis

Kolmogorov-Smirnov analysis of the convergence of a time-series average: for each candidate
equilibration cut, a two-sample KS test between the halves of the remaining series gives an error
estimate, from which the equilibration time, the error of the average and the time needed to get
below a target error are derived. Method write-up: `method_summary.pdf`.
Author: Martin Stroet (University of Queensland). Licence: `LICENSE`.

Status: live-support library (imported by `gromos_job_wrapper` and `atb_condensed_phase`); the
method is stable, the package itself is untouched since its Python 3 conversion.

## Install

Python 3; `numpy`, `scipy`, `matplotlib` (see `setup.cfg`). Installed editable in the platform venv
`/home/atb/ATB/.venv`; elsewhere `pip install -e .`. No env vars, hosts or vendor binaries.

## Python API

```python
from ks_convergence_analysis.convergence_analysis import ks_convergence_analysis
err, t_equil, t_below, err_all, fig = ks_convergence_analysis(
    x, y, converged_error_threshold, step_size_in_percent=1, nsigma=1,
    multithread=False, produce_figure=False)
```
Returns the KS error estimate after discarding equilibration, the equilibration time (in `x`
units), the time after which the error is below the threshold (0 if already below at the start),
the error estimate for the whole series, and a figure (or `None`). `multithread=True` (the default)
uses a `multiprocessing.Pool`; pass `False` on shared hosts. `axes=[ax_summary, ax_ks]` plots
into existing axes.

## Command line

No console script is installed:

```bash
python -m ks_convergence_analysis.convergence_analysis -d DATA -t TARGET_ERROR [-p [PLOT]] [-s SIGFIGS] [-v]
```
`DATA` is two whitespace-separated columns `x y` (`#`/`@` comment lines skipped, GROMACS `.xvg`
style). `-p` saves a figure (default name `ks_convergence.png`). Stdout ends with the error
estimate. Example data and calls: `src/ks_convergence_analysis/example/example.sh` and
`src/ks_convergence_analysis/test/data/` (the script's relative paths assume you run it from a
checkout layout without `src/`; adapt to the `python -m` form above).

## Platform fit

Used by `gromos_job_wrapper` (`helpers/data_processing.py`, KS error analysis) and
`atb_condensed_phase` (`analysis/estimators.py`, equilibration discard). It imports no in-house
package itself; `block_averaging` and `mspyplot` appear only in `test/extended_tests.py`.

## Tests

`src/ks_convergence_analysis/test/test.py` and `extended_tests.py` are scripts that regenerate
figures (`*.png` beside them) and compare against block averaging; they are not assertion suites
and `extended_tests.py` needs uninstalled `red_noise`, `image_concat` and `mspyplot`. There is no
pytest suite.
