#!/usr/bin/env python
"""Regenerate the golden reference data used by the test suite.

Two files are produced under ``tests/golden/``:

``fortran_reference.npz``
    Outputs of the *original* Okada (1985, 1992) FORTRAN subroutines for a fixed,
    deterministic set of inputs.  This is the ground truth for
    ``test_fortran_reference.py``.

    Regenerating it requires ``gfortran`` and the original NIED FORTRAN sources
    in ``tests/fortran/``, which are *not* distributed with this repository --
    the permission obtained from NIED covers this PyTorch port, not redistribution
    of the originals.  The frozen ``.npz`` is what the test suite uses, so the
    tests need neither the sources nor a FORTRAN compiler.

``wrapper_snapshot.npz``
    Outputs of the *current* ``OkadaWrapper`` for a fixed set of well-conditioned
    configurations.  This is a characterisation snapshot: it does not prove
    correctness (that is what ``fortran_reference.npz`` is for), it protects the
    wrapper layer -- coordinate rotation, ``fault_origin`` handling, the free
    surface relations -- from unintended changes during refactoring.

Run this only when the reference itself must change, and review the diff.

    python tests/generate_golden.py
"""
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FORTRAN_DIR = os.path.join(HERE, "fortran")
BUILD_DIR = os.path.join(FORTRAN_DIR, "build")
GOLDEN_DIR = os.path.join(HERE, "golden")

SEED = 20260907
N_REGULAR = 200
N_DEGENERATE = 200

# number of trailing arguments actually consumed by each MODE of driver.f
NARG = {1: 9, 2: 11, 3: 10, 4: 13}
NOUT = {1: 9, 2: 9, 3: 12, 4: 12}
MODE_NAME = {1: "SPOINT", 2: "SRECTF", 3: "DC3D0", 4: "DC3D"}


# --------------------------------------------------------------------------
# FORTRAN reference
# --------------------------------------------------------------------------
def build_driver():
    """Compile driver.f + the original subroutines into a double-precision binary.

    ``DC3D0``/``DC3D`` declare their public interface as ``REAL*4``.  Those
    declarations are stripped so that the whole reference runs in double
    precision (``IMPLICIT REAL*8`` then applies), which lets the test suite
    compare against OkadaTorch at full float64 accuracy.
    """
    if shutil.which("gfortran") is None:
        raise RuntimeError("gfortran not found; cannot regenerate the FORTRAN golden data")
    os.makedirs(BUILD_DIR, exist_ok=True)

    for name in ("okada1985.f", "okada1992.f"):
        src = open(os.path.join(FORTRAN_DIR, name)).read().split("\n")
        out, skip = [], False
        for line in src:
            if line.startswith("      REAL*4"):
                skip = True
                continue
            if skip and len(line) > 5 and line[:5].strip() == "" and line[5] not in (" ", "0"):
                continue  # continuation line of the REAL*4 declaration
            skip = False
            out.append(line)
        open(os.path.join(BUILD_DIR, name), "w").write("\n".join(out))
    shutil.copy(os.path.join(FORTRAN_DIR, "driver.f"), BUILD_DIR)

    exe = os.path.join(BUILD_DIR, "driver")
    subprocess.run(
        ["gfortran", "-std=legacy", "-O2", "-w", "-o", exe,
         "driver.f", "okada1985.f", "okada1992.f"],
        cwd=BUILD_DIR, check=True,
    )
    return exe


def make_cases(mode, n, degenerate, rng):
    """Deterministic input generation.

    ``degenerate=True`` deliberately produces the configurations that stress the
    singularity guards: exact zeros, vertical dips, stations on fault edges.
    """
    def u(lo, hi):
        return rng.uniform(lo, hi, n)

    def zero_some(v, p):
        if not degenerate:
            return v
        v = v.copy()
        v[rng.random(n) < p] = 0.0
        return v

    if mode in (1, 2):
        alp = u(0.3, 0.7)
        if degenerate:
            dip = rng.choice([90.0, -90.0, 0.0, 89.999999, 45.0, 1e-7], n)
        else:
            dip = u(-90.0, 90.0)
        sd, cd = np.sin(np.deg2rad(dip)), np.cos(np.deg2rad(dip))
        sd = np.where(np.abs(cd) < 1e-6, np.sign(sd), sd)
        cd = np.where(np.abs(cd) < 1e-6, 0.0, cd)
        x, y = zero_some(u(-30, 30), 0.25), zero_some(u(-30, 30), 0.25)
        d = zero_some(np.abs(u(0.0, 20.0)), 0.15) if degenerate else np.abs(u(0.01, 20.0))
        d1, d2, d3 = (zero_some(u(-2, 2), 0.25) for _ in range(3))
        if mode == 1:
            return np.stack([alp, x, y, d, sd, cd, d1, d2, d3], axis=1)
        al, aw = np.abs(u(1, 30)), np.abs(u(1, 30))
        return np.stack([alp, x, y, d, al, aw, sd, cd, d1, d2, d3], axis=1)

    alpha = u(0.5, 0.9)
    x, y = zero_some(u(-30, 30), 0.25), zero_some(u(-30, 30), 0.25)
    z = zero_some(-np.abs(u(0, 20)), 0.30) if degenerate else -np.abs(u(0.01, 20))
    depth = np.abs(u(1, 25))
    dip = rng.choice([90.0, -90.0, 0.0, 45.0, 89.9999999], n) if degenerate else u(-90, 90)
    if mode == 3:
        pot = [zero_some(u(-2, 2), 0.25) for _ in range(4)]
        return np.stack([alpha, x, y, z, depth, dip] + pot, axis=1)

    al1, al2 = -np.abs(u(1, 20)), np.abs(u(1, 20))
    aw1, aw2 = -np.abs(u(1, 20)), np.abs(u(1, 20))
    if degenerate:
        on_edge = rng.random(n) < 0.30
        x = np.where(on_edge, al2, x)
        z = np.where(rng.random(n) < 0.20, 0.0, z)
    disl = [zero_some(u(-2, 2), 0.25) for _ in range(3)]
    return np.stack([alpha, x, y, z, depth, dip, al1, al2, aw1, aw2] + disl, axis=1)


def run_driver(exe, mode, args):
    lines = []
    for row in args:
        padded = list(row) + [0.0] * (16 - len(row))
        lines.append(f"{mode} " + " ".join(repr(float(v)) for v in padded))
    proc = subprocess.run([exe], input="\n".join(lines) + "\n",
                          capture_output=True, text=True, check=True)
    rows = proc.stdout.strip().split("\n")
    if len(rows) != len(args):
        raise RuntimeError(f"driver returned {len(rows)} rows for {len(args)} cases")
    iret = np.array([int(r[:3]) for r in rows], dtype=np.int32)
    out = np.array([[float(t) for t in r[3:].split()] for r in rows])
    return iret, out[:, :NOUT[mode]]


def generate_fortran_golden():
    exe = build_driver()
    rng = np.random.default_rng(SEED)
    data = {}
    for mode in (1, 2, 3, 4):
        for tag, n, deg in (("regular", N_REGULAR, False), ("degenerate", N_DEGENERATE, True)):
            args = make_cases(mode, n, deg, rng)
            iret, out = run_driver(exe, mode, args)
            key = f"{MODE_NAME[mode]}_{tag}"
            data[key + "_args"] = args
            data[key + "_iret"] = iret
            data[key + "_out"] = out
            n_bad = int((iret != 0).sum())
            n_nan = int((~np.isfinite(out)).any(axis=1).sum())
            print(f"  {key:<22} {n:4d} cases   IRET!=0: {n_bad:3d}   non-finite: {n_nan:3d}")
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    path = os.path.join(GOLDEN_DIR, "fortran_reference.npz")
    np.savez_compressed(path, **data)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# OkadaWrapper characterisation snapshot
# --------------------------------------------------------------------------
def wrapper_configurations():
    """Well-conditioned configurations only.

    Singular geometries are deliberately excluded: the recommended fixes
    (returning 0 instead of 1e6 in RRX, zeroing the output when IRET != 0) are
    *meant* to change those values, so freezing them here would fight the fixes.
    """
    import torch

    x = torch.tensor([-137.0, -41.0, 13.0, 88.0, 211.0], dtype=torch.float64)
    y = torch.tensor([203.0, -66.0, 7.0, -155.0, 91.0], dtype=torch.float64)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    Z = torch.full_like(X, -7.0)

    def p(**over):
        base = dict(x_fault=3.0, y_fault=-11.0, depth=6.5, strike=189.0,
                    dip=57.0, rake=101.0, slip=5.62)
        base.update(over)
        return {k: torch.tensor(v, dtype=torch.float64) for k, v in base.items()}

    rect = dict(length=218.0, width=46.0)
    cases = {}
    for origin in ("topleft", "center"):
        for strain in (False, True):
            cases[f"rect_surface_{origin}_strain{int(strain)}"] = (
                {"x": X, "y": Y}, p(**rect), dict(fault_origin=origin, compute_strain=strain))
            cases[f"rect_depth_{origin}_strain{int(strain)}"] = (
                {"x": X, "y": Y, "z": Z}, p(**rect), dict(fault_origin=origin, compute_strain=strain))
    for strain in (False, True):
        cases[f"point_surface_strain{int(strain)}"] = (
            {"x": X, "y": Y}, p(), dict(compute_strain=strain))
        cases[f"point_depth_strain{int(strain)}"] = (
            {"x": X, "y": Y, "z": Z}, p(), dict(compute_strain=strain))
    # a vertical fault and a radian-input case, to pin those code paths too
    cases["rect_surface_dip90"] = (
        {"x": X, "y": Y}, p(dip=90.0, **rect), dict(compute_strain=True))
    cases["rect_depth_nu030"] = (
        {"x": X, "y": Y, "z": Z}, p(**rect), dict(compute_strain=True, nu=0.30))
    return cases


def generate_wrapper_snapshot():
    import torch
    from OkadaTorch import OkadaWrapper

    okada = OkadaWrapper()
    data = {}
    for name, (coords, params, kw) in wrapper_configurations().items():
        out = okada.compute(coords, params, **kw)
        data[name] = torch.stack(out).detach().numpy()
        print(f"  {name:<34} {data[name].shape}")
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    path = os.path.join(GOLDEN_DIR, "wrapper_snapshot.npz")
    np.savez_compressed(path, **data)
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(HERE))
    print("== FORTRAN reference ==")
    generate_fortran_golden()
    print("== OkadaWrapper snapshot ==")
    generate_wrapper_snapshot()
