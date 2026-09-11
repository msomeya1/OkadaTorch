# OkadaTorch test suite

```shell
pip install pytest
pytest                 # ~1 min; runs on CPU and, if present, cuda:1
pytest -m "not slow" -q
pytest tests/test_gradients.py -k rake      # one topic
```

Current status: **420 passed** (PyTorch 2.7.1 + CUDA, A100).
On a machine without a GPU, 232 pass and 12 skip; that is what CI runs.

## Layout

| file | tests | what it protects |
|---|---:|---|
| `test_fortran_reference.py` | 21 | The ported formulae reproduce the original Okada (1985, 1992) FORTRAN, on 1600 fixed inputs. This is the ground truth. |
| `test_cross_consistency.py` | 46 | Relations that must hold *between* code paths: 1985 vs 1992 at z=0, analytic strain vs autodiff, free-surface conditions, `topleft` vs `center`, degree vs radian, linearity in slip. Plus a characterisation snapshot of `OkadaWrapper`. |
| `test_gradients.py` | 68 | AD gradients agree with central finite differences, for every parameter, on both the surface and the depth path. |
| `test_singular_behaviour.py` | 135 | Values and gradients stay finite at ordinary geometries; singular geometries behave like the original. |
| `test_api.py` | 133 | Shapes, dtypes, devices, immutability of the inputs, and input validation. |
| `test_performance_contract.py` | 17 | Structural properties the speed rests on: one graph with no breaks, no host-device synchronisation, a bounded aten operation count, `vmap` over source parameters. |

`test_fortran_reference.py` is what proves the maths is right. Everything else
exists because a single reference test cannot see the layers above the formulae
(the wrapper's coordinate rotation, autodiff, the singularity guards) -- and
those are exactly where the known bugs live.

## Golden data

`golden/fortran_reference.npz`
: Outputs of the original FORTRAN for 1600 deterministic inputs
  (200 regular + 200 degenerate per subroutine, seed 20260907). The degenerate
  set deliberately contains exact zeros, vertical dips and stations on fault
  edges. **gfortran is not needed to run the tests** -- only to regenerate.

`golden/wrapper_snapshot.npz`
: Outputs of the *current* `OkadaWrapper` for 14 well-conditioned
  configurations. This does not prove correctness; it detects drift during
  refactoring. Singular geometries are deliberately excluded, because several
  planned fixes are *meant* to change those values.

To regenerate (only when the reference itself must change -- review the diff):

```shell
python tests/generate_golden.py
```

The generator strips the `REAL*4` interface declarations from `okada1992.f` so
that the reference runs entirely in double precision; otherwise the comparison
would be capped at single-precision accuracy.

## Continuous integration

`.github/workflows/test.yml` runs the suite on every push and pull request, on
Python 3.9 (the floor declared in `pyproject.toml`) and 3.13. Nothing has to be
done by hand: a tick or a cross appears on the commit.

The runners have no GPU, so the CUDA half skips itself. What CI does cover, and
a development machine does not, is the **clean install**: it builds the package
from `pyproject.toml` alone, which is the only way to notice a dependency that
happens to already be present locally.

Verified before the workflow was committed: Python 3.9 + torch 2.8.0 and Python
3.13 + torch 2.14.0 both give 232 passed, 12 skipped from a fresh virtualenv.

## Performance contract

`test_performance_contract.py` does not measure any wall-clock time -- timings
on a shared runner are too noisy to gate on. It pins the *structural* properties
that the speed rests on:

- `compute` traces to **one** graph with **no breaks**, so `torch.compile` can
  fuse it;
- it performs **no host-device synchronisation**, so `mode="reduce-overhead"`
  can capture it as a CUDA graph;
- the aten operation count stays within a budget.

All three are invisible in ordinary use: a single `if some_tensor != 0:`
reintroduces a graph break *and* a synchronisation, the answers stay correct,
and the code quietly becomes an order of magnitude slower. That exact line was
put back as a check -- both the graph-break test and the synchronisation test
caught it.

If a deliberate change pushes an operation count past its budget, re-measure and
raise the budget in the same commit, so the new cost is recorded rather than
hidden.

## `tests/fortran/`

Contains the original NIED FORTRAN sources, copied verbatim, used only by
`generate_golden.py`. They are **not** needed to run the suite.

> Note: the repository README states that permission was obtained from NIED to
> publish the *ported* programs. Redistributing the original `.f` sources is a
> separate question. If that is not desired, delete `tests/fortran/` before
> publishing and keep `golden/fortran_reference.npz` -- the suite still runs, and
> only regeneration becomes unavailable.
