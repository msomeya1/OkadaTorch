"""Behaviour at and near the singularities of the Okada solution.

Two distinct concerns:

1. *Values*.  The original FORTRAN detects singular geometries and returns zeros
   with ``IRET != 0``; OkadaTorch raises the flag but keeps computing, so callers
   see NaN (B-1, B-2).
2. *Gradients*.  ``torch.where`` evaluates both branches, so the discarded branch
   can contribute ``0 * inf = NaN`` to the gradient even at perfectly ordinary
   stations (A-2).

Concern 2 is the dangerous one: the forward pass is correct, so nothing looks
wrong until an optimiser silently stops moving.
"""
import numpy as np
import pytest
import torch

from OkadaTorch import DC3D, SPOINT, SRECTF

from conftest import PARAM_NAMES, make_coords, make_params


def _finite(tensors):
    return all(bool(torch.isfinite(t).all()) for t in tensors)


# ---------------------------------------------------------------------------
# ordinary geometries must be clean, in value *and* in gradient
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_regular_geometry_gives_finite_values(okada, device, with_z):
    out = okada.compute(make_coords(device, with_z=with_z),
                        make_params(device), compute_strain=True)
    assert _finite(out)


@pytest.mark.parametrize("name", PARAM_NAMES)
@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_regular_geometry_gives_finite_gradients(okada, device, name, with_z):
    coords = make_coords(device, with_z=with_z)
    p = {k: v.clone().detach().requires_grad_(k == name)
         for k, v in make_params(device).items()}
    ux, uy, uz = okada.compute(coords, p, compute_strain=False)
    g = torch.autograd.grad((ux + uy + uz).sum(), p[name])[0]
    assert torch.isfinite(g).all(), f"d/d({name}) is not finite for a regular geometry"


# ---------------------------------------------------------------------------
# known-bad: A-2, B-1, B-2
# ---------------------------------------------------------------------------
def _on_fault_edge(device):
    """A station sitting exactly on the top edge of the fault at z=0."""
    coords = {"x": torch.tensor([0.0, 2.0], device=device, dtype=torch.float64),
              "y": torch.tensor([0.0, 0.0], device=device, dtype=torch.float64),
              "z": torch.tensor([0.0, 0.0], device=device, dtype=torch.float64)}
    params = make_params(device, x_fault=0.0, y_fault=0.0, depth=0.0,
                         strike=0.0, dip=45.0, rake=90.0, slip=1.0,
                         length=10.0, width=5.0)
    return coords, params


# --- A-2 regression: stations that trip a singularity *guard* without being
# --- singular themselves.  The forward pass is finite at all of them; before the
# --- denominators were sanitised, torch.where still evaluated the discarded
# --- branch and 0*inf reached the backward pass.
GUARD_TRIPPING_GEOMETRIES = {
    # xi == 0 exactly (station on the along-strike axis) with a vertical fault,
    # so both the XI==0 guard and the CD==0 branch are active at once
    "vertical_fault_xi0": (
        dict(x=[0.0, 4.0], y=[8.0, 15.0], z=[-2.0, -2.0]),
        dict(x_fault=0.0, y_fault=0.0, depth=6.0, strike=0.0, dip=90.0,
             rake=45.0, slip=3.0, length=25.0, width=10.0)),
    # station on the along-strike axis of an inclined fault (XI==0 guard in A5/AI4)
    "inclined_fault_xi0": (
        dict(x=[0.0, 6.0], y=[12.0, 20.0]),
        dict(x_fault=0.0, y_fault=0.0, depth=4.0, strike=0.0, dip=50.0,
             rake=70.0, slip=2.0, length=30.0, width=12.0)),
    # far down-dip, where the R+eta guard (KET / RET==0) engages
    "down_dip_extension": (
        dict(x=[2.0, 5.0], y=[-40.0, -60.0], z=[-3.0, -3.0]),
        dict(x_fault=0.0, y_fault=0.0, depth=5.0, strike=0.0, dip=30.0,
             rake=90.0, slip=1.0, length=20.0, width=15.0)),
}


def _geometry(name, device):
    c, p = GUARD_TRIPPING_GEOMETRIES[name]
    coords = {k: torch.tensor(v, device=device, dtype=torch.float64) for k, v in c.items()}
    params = {k: torch.tensor(v, device=device, dtype=torch.float64) for k, v in p.items()}
    return coords, params


@pytest.mark.parametrize("geometry", sorted(GUARD_TRIPPING_GEOMETRIES))
@pytest.mark.parametrize("name", PARAM_NAMES)
def test_gradient_is_finite_where_a_singularity_guard_engages(okada, device, geometry, name):
    coords, params0 = _geometry(geometry, device)
    assert _finite(okada.compute(coords, params0, compute_strain=False)), \
        "precondition: this geometry must have a finite forward pass"
    p = {k: v.clone().detach().requires_grad_(k == name) for k, v in params0.items()}
    ux, uy, uz = okada.compute(coords, p, compute_strain=False)
    g = torch.autograd.grad((ux + uy + uz).sum(), p[name], allow_unused=True)[0]
    assert g is not None and torch.isfinite(g).all(), \
        f"d/d({name}) is NaN/Inf at {geometry} although the forward pass is finite"


@pytest.mark.parametrize("geometry", sorted(GUARD_TRIPPING_GEOMETRIES))
def test_coordinate_gradient_is_finite_where_a_guard_engages(okada, device, geometry):
    coords, params = _geometry(geometry, device)
    assert _finite(okada.gradient(coords, params, arg="x", compute_strain=False))


def test_fault_edge_station_is_usable(okada):
    """B-1 regression.  A station sitting exactly on the fault edge is flagged,
    zeroed, and -- because the flagged stations are fed a dummy geometry rather
    than merely masked afterwards -- differentiable."""
    coords, params0 = _on_fault_edge(torch.device("cpu"))
    assert _finite(okada.compute(coords, params0, compute_strain=False))
    for name in ("depth", "dip", "slip", "x_fault"):
        p = {k: v.clone().detach().requires_grad_(k == name) for k, v in params0.items()}
        ux, uy, uz = okada.compute(coords, p, compute_strain=False)
        g = torch.autograd.grad((ux + uy + uz).sum(), p[name], allow_unused=True)[0]
        assert g is not None and torch.isfinite(g).all()


def test_singular_stations_return_zero_like_the_original():
    device = torch.device("cpu")
    x = torch.tensor([0.0, 0.0, 5.0], device=device, dtype=torch.float64)
    y = torch.tensor([0.0, 1e-9, 0.0], device=device, dtype=torch.float64)
    z = torch.zeros_like(x)
    out, iret = DC3D(2.0 / 3.0, x, y, z,
                     torch.tensor(0.0, dtype=torch.float64),
                     torch.tensor(45.0, dtype=torch.float64),
                     0.0, 10.0, -5.0, 0.0, 1.0, 0.0, 0.0,
                     compute_strain=False, is_degree=True)
    sing = iret != 0
    assert sing.any(), "test geometry no longer triggers IRET"
    for t in out:
        assert torch.all(t[sing] == 0.0), "singular stations must return 0, not NaN"


def test_positive_z_returns_zero_like_the_original():
    out, iret = DC3D(2.0 / 3.0,
                     torch.tensor([1.0], dtype=torch.float64),
                     torch.tensor([1.0], dtype=torch.float64),
                     torch.tensor([2.0], dtype=torch.float64),   # z > 0
                     torch.tensor(5.0, dtype=torch.float64),
                     torch.tensor(45.0, dtype=torch.float64),
                     0.0, 10.0, -5.0, 0.0, 1.0, 0.0, 0.0,
                     compute_strain=False, is_degree=True)
    assert bool((iret == 2).all())
    for t in out:
        assert torch.all(t == 0.0)


def test_rrx_singularity_returns_zero_like_the_1992_kernel(okada):
    """B-4 regression.  RRX = 1/(R(R+xi)) used to fall back to a constant of
    order 1e6, whose meaning changed with the choice of metres or kilometres.
    The 1992 routines set the identical quantity (DCCON2's X11) to zero, so that
    is what is used now."""
    import inspect
    import OkadaTorch.utils as utils
    code = "\n".join(line for line in inspect.getsource(utils._SRECTG).split("\n")
                     if not line.strip().startswith("#"))
    assert "1.0e6" not in code and "1e6" not in code

    # and the choice is unobservable: where the fallback fires, Q = Y = D = 0
    T = lambda v: torch.tensor(v, dtype=torch.float64)
    out = SRECTF(0.5, T([-8.0]), T([0.0]), T(0.0), T(20.0), T(10.0),
                 T(1.0), T(0.0), 1.0, 0.5, 0.3, compute_strain=True)
    assert _finite(out)


# ---------------------------------------------------------------------------
# Characterisation of the branchless rewrite (DEBUG_NOTES.md A-1).
#
# Removing `if DISLn != 0.0` means every contribution is now evaluated even when
# its dislocation is zero.  At geometries where the Okada kernel itself is
# singular -- R = 0, R + d = 0, R + xi = 0, R + eta = 0 -- the corresponding
# expression is inf, and `0 * inf` is NaN.  The original FORTRAN returned a
# finite number there, but only by accident: it skipped the block because that
# particular dislocation happened to be exactly zero.
#
# A randomised sweep of 4000 deliberately degenerate cases per subroutine found
# that *every* such difference sits on one of those four singular loci, and that
# values are bit-identical everywhere else.  B-1 (zeroing the output when the
# geometry is singular) is the proper fix.
# ---------------------------------------------------------------------------
def test_up_dip_extension_of_a_vertical_fault_is_finite():
    """Vertical fault, Q = 0, xi = 0, eta = -2: R = 2 and d = -2, so R + d = 0.
    Only the dip-slip component is non-zero, so the physically correct answer is
    the finite dip-slip contribution alone."""
    out = SRECTF(0.5,
                 torch.tensor(0.0, dtype=torch.float64),   # X
                 torch.tensor(0.0, dtype=torch.float64),   # Y  -> Q = 0
                 torch.tensor(3.0, dtype=torch.float64),   # DEP
                 torch.tensor(10.0, dtype=torch.float64),  # AL
                 torch.tensor(5.0, dtype=torch.float64),   # AW -> eta = -2
                 torch.tensor(1.0, dtype=torch.float64),   # SD  (vertical)
                 torch.tensor(0.0, dtype=torch.float64),   # CD
                 0.0, 1.0, 0.0,                            # strike-slip is zero
                 compute_strain=True)
    assert _finite(out)


# ---------------------------------------------------------------------------
# B-1: reporting.  Zeroing alone would make a singular station indistinguishable
# from one whose displacement is genuinely zero, so the flag has to be reachable.
# ---------------------------------------------------------------------------
def _mixed_stations(device):
    """Three stations: on a fault corner, above the surface, and ordinary."""
    coords = {"x": torch.tensor([0.0, 1.0, 7.0], device=device, dtype=torch.float64),
              "y": torch.tensor([0.0, 1.0, 9.0], device=device, dtype=torch.float64),
              "z": torch.tensor([0.0, 2.0, -2.0], device=device, dtype=torch.float64)}
    params = make_params(device, x_fault=0.0, y_fault=0.0, depth=0.0, strike=0.0,
                         dip=45.0, rake=90.0, slip=1.0, length=10.0, width=5.0)
    return coords, params


def test_return_iret_is_off_by_default(okada, device):
    """Existing callers must keep getting a plain list back."""
    out = okada.compute(*_mixed_stations(device), compute_strain=False)
    assert isinstance(out, list) and len(out) == 3


@pytest.mark.parametrize("strain", [False, True])
def test_return_iret_flags_and_zeroes_bad_stations(okada, device, strain):
    coords, params = _mixed_stations(device)
    out, iret = okada.compute(coords, params, compute_strain=strain, return_iret=True)
    assert iret.shape == coords["x"].shape
    assert iret.tolist() == [1, 2, 0], "expected singular / above-surface / normal"
    bad = iret != 0
    for t in out:
        assert torch.all(t[bad] == 0.0)
        assert torch.isfinite(t).all()


def test_iret_distinguishes_singular_from_genuinely_zero(okada, device):
    """The whole point of exposing the flag: `u == 0` is ambiguous on its own."""
    coords, params = _mixed_stations(device)
    _, iret_singular = okada.compute(coords, params, compute_strain=False, return_iret=True)
    params_zero = dict(params)
    params_zero["slip"] = torch.zeros_like(params["slip"])
    out, iret_zero = okada.compute(coords, params_zero, compute_strain=False, return_iret=True)
    assert torch.all(torch.stack(out) == 0.0), "zero slip must give zero displacement"
    assert iret_zero.tolist() == iret_singular.tolist(), \
        "IRET describes the geometry, not the source strength"


def test_surface_and_depth_paths_agree_on_which_stations_are_singular(okada, device):
    """D4: SPOINT/SRECTF have no return code in the original FORTRAN.  Having
    added one, the 1985 and the 1992 formulation must classify a station at z=0
    identically -- otherwise passing `z=0` explicitly would change the answer."""
    coords, params = _mixed_stations(device)
    coords["z"] = torch.zeros_like(coords["x"])
    _, iret_depth = okada.compute(coords, params, compute_strain=False, return_iret=True)
    surface = {"x": coords["x"], "y": coords["y"]}
    _, iret_surface = okada.compute(surface, params, compute_strain=False, return_iret=True)
    assert iret_surface.tolist() == iret_depth.tolist()


@pytest.mark.parametrize("rect", [False, True], ids=["point", "rect"])
def test_low_level_1985_routines_keep_their_old_signature(rect):
    """`return_iret` defaults to False so SPOINT/SRECTF still return just a list."""
    T = lambda v: torch.tensor(v, dtype=torch.float64)
    if rect:
        out = SRECTF(0.5, T([1.0]), T([2.0]), T(3.0), T(10.0), T(5.0),
                     T(0.7071067811865476), T(0.7071067811865476), 1.0, 0.0, 0.0,
                     compute_strain=False)
    else:
        out = SPOINT(0.5, T([1.0]), T([2.0]), T(3.0),
                     T(0.7071067811865476), T(0.7071067811865476), 1.0, 0.0, 0.0,
                     compute_strain=False)
    assert isinstance(out, list) and len(out) == 3


@pytest.mark.parametrize("name", PARAM_NAMES)
def test_gradients_stay_finite_when_a_station_is_singular(okada, device, name):
    """Zeroing the output is not enough on its own: torch.where hands zero back
    to the discarded branch, which then evaluates 0 * d(nan)/dx.  The flagged
    stations must be fed a dummy geometry so the whole graph stays finite."""
    coords, params0 = _mixed_stations(device)
    p = {k: v.clone().detach().requires_grad_(k == name) for k, v in params0.items()}
    ux, uy, uz = okada.compute(coords, p, compute_strain=False)
    g = torch.autograd.grad((ux + uy + uz).sum(), p[name], allow_unused=True)[0]
    assert g is not None and torch.isfinite(g).all(), \
        f"d/d({name}) is not finite even though every station is either usable or zeroed"
