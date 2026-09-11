"""Internal consistency of OkadaWrapper.

These are independent of the FORTRAN golden data: they check relations that must
hold between different code paths, which catches wiring mistakes (wrong rotation,
wrong ``fault_origin`` offset, swapped strain components) that a single-path
reference test cannot.
"""
import numpy as np
import pytest
import torch

from conftest import make_coords, make_params, rel_err

# Okada 1985 (surface) and Okada 1992 (z=0) are different formulae for the same
# quantity, so agreement is a genuine cross-check rather than a tautology.
TOL_1985_VS_1992 = 1e-9
# analytic strain vs. autodiff of the displacement: limited only by float64
TOL_STRAIN_VS_AD = 1e-10


@pytest.mark.parametrize("rect", [False, True], ids=["point", "rect"])
@pytest.mark.parametrize("origin", ["topleft", "center"])
def test_surface_1985_matches_depth_1992_at_z0(okada, device, rect, origin):
    coords = make_coords(device)
    params = make_params(device, rect=rect)
    coords_z = dict(coords, z=torch.zeros_like(coords["x"]))

    a = okada.compute(coords, params, compute_strain=True, fault_origin=origin)
    b = okada.compute(coords_z, params, compute_strain=True, fault_origin=origin)

    names = "ux uy uz uxx uyx uzx uxy uyy uzy uxz uyz uzz".split()
    for n, u, v in zip(names, a, b):
        err = rel_err(u.detach().cpu().numpy(), v.detach().cpu().numpy()).max()
        assert err < TOL_1985_VS_1992, f"{n}: 1985 vs 1992 differ by {err:.3e}"


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_analytic_strain_matches_autodiff(okada, device, with_z):
    """The 9 spatial derivatives returned by compute() must equal d(u)/d(x,y,z)."""
    coords = make_coords(device, with_z=with_z)
    params = make_params(device)
    full = okada.compute(coords, params, compute_strain=True)

    args = ["x", "y", "z"] if with_z else ["x", "y"]
    for j, arg in enumerate(args):
        grad = okada.gradient(coords, params, arg=arg, compute_strain=False)
        for i in range(3):                      # ux, uy, uz
            analytic = full[3 + 3 * j + i]      # u{i}{arg}
            err = rel_err(analytic.detach().cpu().numpy(),
                          grad[i].detach().cpu().numpy()).max()
            assert err < TOL_STRAIN_VS_AD, \
                f"d(u{i})/d{arg}: analytic vs autodiff differ by {err:.3e}"


def test_free_surface_conditions_hold(okada, device):
    """At z=0 the traction-free surface forces uxz=-uzx, uyz=-uzy,
    uzz = -nu/(1-nu) (uxx+uyy).  The 1992 path computes these independently,
    so this pins the physics rather than the implementation."""
    nu = 0.25
    coords = dict(make_coords(device))
    coords["z"] = torch.zeros_like(coords["x"])
    out = okada.compute(coords, make_params(device), compute_strain=True, nu=nu)
    ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz = out
    n = lambda t: t.detach().cpu().numpy()
    assert rel_err(n(uxz), -n(uzx)).max() < 1e-9
    assert rel_err(n(uyz), -n(uzy)).max() < 1e-9
    assert rel_err(n(uzz), -(n(uxx) + n(uyy)) * nu / (1 - nu)).max() < 1e-9


def test_degree_and_radian_inputs_agree(okada, device):
    coords = make_coords(device, with_z=True)
    deg = make_params(device)
    rad = {k: (v * torch.pi / 180.0 if k in ("strike", "dip", "rake") else v)
           for k, v in deg.items()}
    a = okada.compute(coords, deg, compute_strain=True, is_degree=True)
    b = okada.compute(coords, rad, compute_strain=True, is_degree=False)
    for u, v in zip(a, b):
        assert rel_err(u.detach().cpu().numpy(), v.detach().cpu().numpy()).max() < 1e-9


def test_topleft_and_center_describe_the_same_fault(okada, device):
    """Shifting the reference point from the top-left corner to the centre and
    compensating in x_fault/y_fault/depth must give identical displacements.
    This is the only test that would catch a sign error in the fault_origin
    offsets, which the FORTRAN reference cannot see."""
    coords = make_coords(device, with_z=True)
    p = make_params(device)
    L, W = p["length"], p["width"]
    ss = torch.sin(torch.deg2rad(p["strike"]))
    cs = torch.cos(torch.deg2rad(p["strike"]))
    sd = torch.sin(torch.deg2rad(p["dip"]))
    cd = torch.cos(torch.deg2rad(p["dip"]))

    # Move the reference point from the top-left corner to the centre:
    # L/2 along strike, W/2 down dip.  In the fault-local frame (x' along strike)
    # a down-dip step of s moves the reference by (0, -s*cd) and deepens it by
    # s*sd; rotating that offset into ENU gives the expressions below.
    along, down_dip = L / 2, W / 2
    dxl, dyl = along, -down_dip * cd
    p_center = dict(p)
    p_center["x_fault"] = p["x_fault"] + dxl * ss - dyl * cs
    p_center["y_fault"] = p["y_fault"] + dxl * cs + dyl * ss
    p_center["depth"] = p["depth"] + down_dip * sd

    a = okada.compute(coords, p, compute_strain=False, fault_origin="topleft")
    b = okada.compute(coords, p_center, compute_strain=False, fault_origin="center")
    for u, v in zip(a, b):
        err = rel_err(u.detach().cpu().numpy(), v.detach().cpu().numpy()).max()
        assert err < 1e-9, f"topleft/center describe different faults: {err:.3e}"


def test_zero_slip_gives_zero_displacement(okada, device):
    out = okada.compute(make_coords(device, with_z=True),
                        make_params(device, slip=0.0), compute_strain=True)
    for t in out:
        assert torch.all(t == 0.0)


def test_displacement_scales_linearly_with_slip(okada, device):
    coords = make_coords(device, with_z=True)
    a = okada.compute(coords, make_params(device, slip=1.0), compute_strain=True)
    b = okada.compute(coords, make_params(device, slip=3.5), compute_strain=True)
    for u, v in zip(a, b):
        err = rel_err((u * 3.5).detach().cpu().numpy(), v.detach().cpu().numpy()).max()
        assert err < 1e-12


class TestWrapperSnapshot:
    """Characterisation tests: freeze today's wrapper output so that refactoring
    cannot silently change it.  Correctness is established elsewhere; this only
    detects *drift*.  Regenerate deliberately with tests/generate_golden.py."""

    def test_all_configurations_unchanged(self, okada, wrapper_golden, device):
        from generate_golden import wrapper_configurations
        for name, (coords, params, kw) in wrapper_configurations().items():
            coords = {k: v.to(device) for k, v in coords.items()}
            params = {k: v.to(device) for k, v in params.items()}
            got = torch.stack(okada.compute(coords, params, **kw)).detach().cpu().numpy()
            err = rel_err(got, wrapper_golden[name]).max()
            assert err < 1e-11, f"{name}: wrapper output drifted by {err:.3e}"


def test_surrogate_gradient_does_not_perturb_the_value():
    """A-3 uses a straight-through estimator near a vertical fault.  It must be
    bit-exact in the forward direction -- the surrogate can be orders of
    magnitude larger than the value it stands in for, so a naive spelling would
    leave a visible rounding error."""
    from OkadaTorch.utils import _surrogate_grad
    value = torch.tensor([1.0, 2.5, -3.75e-8], dtype=torch.float64)
    surrogate = torch.tensor([1e9, -4e7, 6.0], dtype=torch.float64, requires_grad=True)
    use = torch.tensor([True, True, False])
    out = _surrogate_grad(use, value, surrogate)
    assert torch.equal(out, value), "the straight-through estimator changed the value"
    out.sum().backward()
    assert surrogate.grad.tolist() == [1.0, 1.0, 0.0], \
        "the gradient must come from the surrogate exactly where it is used"


# ---------------------------------------------------------------------------
# D-1: the answer must not depend on the unit the caller happens to use
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("unit_ratio", [1000.0, 0.001], ids=["km->m", "km->Mm"])
@pytest.mark.parametrize("offset", [1.0, 1e-3, 1e-5, 1e-7, 1e-9],
                         ids=lambda v: f"offset{v:g}")
# Below about 1e-8 of the fault length the offset is not representable to any
# useful precision -- `(20 + 1e-9) - 20` in float64 keeps only ~6 digits -- so
# the *value* cannot be expected to agree between unit systems.  The
# classification still must, and that is the part D-1 was about.
def test_result_is_invariant_under_a_change_of_length_unit(okada, device, unit_ratio,
                                                           offset):
    """Every length scaled by the same factor must scale the displacement by that
    factor and nothing else.

    The thresholds used to be absolute (`|xi| < 1e-6`), so the same physical
    configuration crossed them in one unit system and not in the other: a station
    1e-7 km past a fault edge was flagged singular when written in kilometres and
    returned a finite value when written in metres.
    """
    def run(scale):
        # strike = 90 deg puts the along-strike axis along x
        coords = {"x": torch.tensor([(20.0 + offset) * scale, 5.0 * scale],
                                    device=device, dtype=torch.float64),
                  "y": torch.tensor([0.0, 3.0 * scale], device=device,
                                    dtype=torch.float64)}
        params = make_params(device, x_fault=0.0, y_fault=0.0, depth=0.0,
                             length=20.0 * scale, width=10.0 * scale,
                             slip=2.0 * scale, strike=90.0, dip=90.0, rake=90.0)
        params["x_fault"] = torch.zeros((), device=device, dtype=torch.float64)
        out, iret = okada.compute(coords, params, compute_strain=False,
                                  return_iret=True)
        return torch.stack(out) / scale, iret

    a, ia = run(1.0)
    b, ib = run(unit_ratio)
    assert ia.tolist() == ib.tolist(), \
        f"the singular/normal classification changed with the unit: {ia} vs {ib}"
    if offset / 20.0 > 1e-8:
        assert torch.allclose(a, b, rtol=1e-9, atol=1e-14), \
            f"the displacement changed with the unit: max|diff| = {(a-b).abs().max():.3e}"


def test_thresholds_follow_the_working_precision():
    """D-1.  A fixed float64-sized relative tolerance would sit below the float32
    resolution and the guards would never fire; tying it to the machine epsilon
    of the dtype in use keeps them meaningful in both."""
    from OkadaTorch.utils import _rel_eps
    eps64 = _rel_eps(torch.zeros((), dtype=torch.float64))
    eps32 = _rel_eps(torch.zeros((), dtype=torch.float32))
    assert eps64 < 1e-10 < eps32
    assert eps32 > torch.finfo(torch.float32).eps
