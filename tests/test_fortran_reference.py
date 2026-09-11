"""Ground truth: OkadaTorch must reproduce the original FORTRAN subroutines.

The golden data in ``tests/golden/fortran_reference.npz`` was produced by
compiling the original Okada (1985, 1992) sources in double precision; see
``tests/generate_golden.py``.  gfortran is *not* needed to run these tests.

Rows where the FORTRAN itself returned a non-finite value, or flagged a
singularity via ``IRET != 0``, are excluded here -- what OkadaTorch should do in
those cases is a separate design question (see B-1 in DEBUG_NOTES.md) and is
covered by ``test_singular_behaviour.py``.
"""
import numpy as np
import pytest
import torch

from OkadaTorch import DC3D, DC3D0, SPOINT, SRECTF

from conftest import rel_err

# max(|got - ref|) / max(|ref|, 1) must stay below this.
# The observed worst case is 2.5e-12 (DC3D, degenerate); 1e-9 leaves headroom
# for library/compiler differences without letting a real regression through.
TOL = 1e-9

CASES = [(name, tag) for name in ("SPOINT", "SRECTF", "DC3D0", "DC3D")
         for tag in ("regular", "degenerate")]


def _call(name, args, device, dtype):
    a = [torch.tensor(float(v), device=device, dtype=dtype) for v in args]
    if name == "SPOINT":
        return SPOINT(*a[:9], compute_strain=True)
    if name == "SRECTF":
        return SRECTF(*a[:11], compute_strain=True)
    if name == "DC3D0":
        return DC3D0(*a[:10], compute_strain=True, is_degree=True)[0]
    return DC3D(*a[:13], compute_strain=True, is_degree=True)[0]


@pytest.mark.parametrize("name,tag", CASES, ids=[f"{n}-{t}" for n, t in CASES])
def test_matches_original_fortran(fortran_golden, name, tag):
    # CPU only: this is a pure-maths check and running all 1600 cases on both
    # devices doubles the suite runtime for no extra coverage.  Device agreement
    # is checked separately by test_gpu_agrees_with_cpu.
    device = torch.device("cpu")
    key = f"{name}_{tag}"
    args = fortran_golden[key + "_args"]
    ref = fortran_golden[key + "_out"]
    iret = fortran_golden[key + "_iret"]

    usable = (iret == 0) & np.isfinite(ref).all(axis=1)
    assert usable.sum() > 0.8 * len(args), "golden data looks degenerate"

    got = np.stack([
        np.array([t.item() for t in _call(name, args[i], device, torch.float64)])
        for i in np.flatnonzero(usable)
    ])
    # Check finiteness separately: rel_err would be NaN here and every comparison
    # against NaN is False, so a NaN result would slip through the tolerance test
    # below unnoticed.
    bad = np.flatnonzero(~np.isfinite(got).all(axis=1))
    if bad.size:
        i = np.flatnonzero(usable)[bad[0]]
        pytest.fail(f"{key}: {bad.size} case(s) are non-finite where the FORTRAN is "
                    f"finite; first at index {i}\n"
                    f"  args = {args[i]}\n"
                    f"  ref  = {ref[i]}\n"
                    f"  got  = {got[bad[0]]}")

    err = rel_err(got, ref[usable])
    worst = err.max()
    if worst >= TOL:
        j = np.unravel_index(err.argmax(), err.shape)
        pytest.fail(f"{key}: max rel.err {worst:.3e} at case {j[0]} component {j[1]}\n"
                    f"  args = {args[usable][j[0]]}\n"
                    f"  ref  = {ref[usable][j[0]][j[1]]!r}\n"
                    f"  got  = {got[j[0]][j[1]]!r}")


@pytest.mark.parametrize("name,tag", CASES, ids=[f"{n}-{t}" for n, t in CASES])
def test_displacement_only_matches_full_call(fortran_golden, name, tag):
    """``compute_strain=False`` must return exactly the first three components."""
    args = fortran_golden[f"{name}_{tag}_args"][:40]
    dev, dt = torch.device("cpu"), torch.float64
    for row in args:
        a = [torch.tensor(float(v), device=dev, dtype=dt) for v in row]
        if name == "SPOINT":
            full, disp = SPOINT(*a[:9], compute_strain=True), SPOINT(*a[:9], compute_strain=False)
        elif name == "SRECTF":
            full, disp = SRECTF(*a[:11], compute_strain=True), SRECTF(*a[:11], compute_strain=False)
        elif name == "DC3D0":
            full = DC3D0(*a[:10], compute_strain=True)[0]
            disp = DC3D0(*a[:10], compute_strain=False)[0]
        else:
            full = DC3D(*a[:13], compute_strain=True)[0]
            disp = DC3D(*a[:13], compute_strain=False)[0]
        assert len(disp) == 3
        for k in range(3):
            f, d = full[k].item(), disp[k].item()
            if np.isfinite(f) or np.isfinite(d):
                assert f == d or (np.isnan(f) and np.isnan(d)), \
                    f"{name}: component {k} differs between compute_strain=True/False"


def test_iret_flags_agree_with_fortran(fortran_golden):
    """The 1992 routines must raise IRET on exactly the cases the original does."""
    device = torch.device("cpu")
    for name, fn in (("DC3D0", DC3D0), ("DC3D", DC3D)):
        for tag in ("regular", "degenerate"):
            key = f"{name}_{tag}"
            args, ref_iret = fortran_golden[key + "_args"], fortran_golden[key + "_iret"]
            got = []
            for row in args:
                a = [torch.tensor(float(v), device=device, dtype=torch.float64) for v in row]
                n = 10 if name == "DC3D0" else 13
                got.append(int(fn(*a[:n], compute_strain=False, is_degree=True)[1].item()))
            got = np.array(got)
            mism = np.flatnonzero(got != ref_iret)
            assert mism.size == 0, (
                f"{key}: IRET differs on {mism.size} cases, first at index {mism[0]} "
                f"(fortran={ref_iret[mism[0]]}, torch={got[mism[0]]})")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("name", ["SPOINT", "SRECTF", "DC3D0", "DC3D"])
def test_gpu_agrees_with_cpu(fortran_golden, name):
    """The same inputs must give the same numbers on GPU as on CPU."""
    idx = 1 if torch.cuda.device_count() > 1 else 0
    gpu = torch.device(f"cuda:{idx}")
    args = fortran_golden[f"{name}_regular_args"][:50]
    for row in args:
        a = np.array([t.item() for t in _call(name, row, torch.device("cpu"), torch.float64)])
        b = np.array([t.item() for t in _call(name, row, gpu, torch.float64)])
        err = rel_err(b, a).max()
        assert err < 1e-12, f"{name}: GPU and CPU differ by {err:.3e}"
