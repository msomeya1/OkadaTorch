"""Autodiff correctness: AD gradients must agree with central finite differences.

This is the only class of test that can detect A-1 and A-3 -- both of them return
a *plausible* value (zero) rather than raising or producing NaN, and both leave
the forward pass untouched, so every other test in this suite passes happily.

Tests marked ``xfail(strict=True)`` document known bugs.  When the bug is fixed
the test will XPASS, which pytest reports as a failure -- that is the signal to
delete the marker.
"""
import numpy as np
import pytest
import torch

from conftest import PARAM_NAMES, make_coords, make_params

# central difference error is O(h^2 f''') + O(eps |f| / h); with h ~ 1e-5 * scale
# and float64 this bottoms out around 1e-8 relative, so 1e-5 is a safe gate.
FD_TOL = 1e-5


def _functional(okada, coords, params, **kw):
    """A generic scalar functional of the output, so that one number carries
    information from all three displacement components at every station."""
    ux, uy, uz = okada.compute(coords, params, compute_strain=False, **kw)
    return (ux + 2.0 * uy + 3.0 * uz).sum()


def _ad_grad(okada, coords, params, name, **kw):
    p = {k: v.clone().detach().requires_grad_(k == name) for k, v in params.items()}
    return torch.autograd.grad(_functional(okada, coords, p, **kw), p[name])[0].item()


def _fd_grad(okada, coords, params, name, **kw):
    v0 = params[name].item()
    h = 1e-5 * max(abs(v0), 1.0)
    out = []
    for s in (+1.0, -1.0):
        p = {k: v.clone().detach() for k, v in params.items()}
        p[name] = torch.full_like(p[name], v0 + s * h)
        out.append(_functional(okada, coords, p, **kw).item())
    return (out[0] - out[1]) / (2.0 * h)


def _assert_close(ad, fd, what):
    scale = max(abs(fd), abs(ad), 1e-12)
    err = abs(ad - fd) / scale
    assert err < FD_TOL, f"{what}: AD={ad!r} vs FD={fd!r} (rel. err {err:.3e})"


@pytest.mark.parametrize("name", PARAM_NAMES)
@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_parameter_gradient_matches_finite_difference(okada, device, name, with_z):
    coords = make_coords(device, with_z=with_z)
    params = make_params(device)
    _assert_close(_ad_grad(okada, coords, params, name),
                  _fd_grad(okada, coords, params, name), f"d/d({name})")


@pytest.mark.parametrize("name", ["depth", "dip", "slip", "strike"])
def test_wrapper_gradient_method_matches_autograd(okada, device, name):
    """OkadaWrapper.gradient (forward-mode over jacfwd) must agree with the
    ordinary reverse-mode gradient of the same quantity."""
    coords = make_coords(device, with_z=True)
    params = make_params(device)
    g = okada.gradient(coords, params, arg=name, compute_strain=False)
    lumped = (g[0] + 2.0 * g[1] + 3.0 * g[2]).sum().item()
    _assert_close(lumped, _ad_grad(okada, coords, params, name), f"gradient(arg={name})")


@pytest.mark.parametrize("name", ["depth", "slip"])
def test_hessian_matches_finite_difference_of_gradient(okada, device, name):
    coords = make_coords(device, with_z=True)
    params = make_params(device)
    h = okada.hessian(coords, params, arg1=name, arg2=name, compute_strain=False)
    ad2 = (h[0] + 2.0 * h[1] + 3.0 * h[2]).sum().item()

    v0, step = params[name].item(), 1e-4 * max(abs(params[name].item()), 1.0)
    vals = []
    for s in (+1.0, -1.0):
        p = {k: v.clone().detach() for k, v in params.items()}
        p[name] = torch.full_like(p[name], v0 + s * step)
        vals.append(_ad_grad(okada, coords, p, name))
    _assert_close(ad2, (vals[0] - vals[1]) / (2.0 * step), f"d2/d({name})2")


# ---------------------------------------------------------------------------
# Known-bad cases.  See DEBUG_NOTES.md sections A-1 and A-3.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rake", [180.0, -180.0, 90.0, -90.0])
def test_gradient_wrt_rake_at_the_cardinal_rakes(okada, rake):
    """Only rake=0 actually triggers A-1.

    sin(180 deg) evaluates to -1.2e-16 and cos(90 deg) to 6.1e-17 -- not zero --
    so those blocks are never skipped and the gradient happens to come out right.
    Pinning that here keeps the xfail below honest about its true scope."""
    device = torch.device("cpu")
    coords, params = make_coords(device), make_params(device, rake=rake)
    _assert_close(_ad_grad(okada, coords, params, "rake"),
                  _fd_grad(okada, coords, params, "rake"), f"d/d(rake) at rake={rake}")


def test_gradient_wrt_rake_at_pure_strike_slip(okada):
    """A-1 regression.  rake=0 makes u_dip exactly 0.0; while the contributions
    were guarded by `if DISLn != 0.0`, that skipped the dip-slip block and lost
    its contribution to d/d(rake) -- which is the dominant term there."""
    device = torch.device("cpu")
    coords, params = make_coords(device), make_params(device, rake=0.0)
    _assert_close(_ad_grad(okada, coords, params, "rake"),
                  _fd_grad(okada, coords, params, "rake"), "d/d(rake) at rake=0")


def test_gradient_wrt_slip_at_zero_slip(okada):
    device = torch.device("cpu")
    coords, params = make_coords(device), make_params(device, slip=0.0)
    _assert_close(_ad_grad(okada, coords, params, "slip"),
                  _fd_grad(okada, coords, params, "slip"), "d/d(slip) at slip=0")


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_gradient_wrt_dip_at_vertical_fault(okada, with_z):
    """A-3 regression.  d(u)/d(dip) is continuous through dip=90 (the vertical
    formula is the analytic limit), so the value at 90 must be close to the value
    just below it.

    A finite difference cannot be used as the reference here: any step crosses
    the |cos dip| < EPS_DIP snap band, so we compare against the AD gradient at
    dip=89.5, which the tests above have already validated."""
    device = torch.device("cpu")
    coords = make_coords(device, with_z=with_z)
    near = _ad_grad(okada, coords, make_params(device, dip=89.5), "dip")
    at90 = _ad_grad(okada, coords, make_params(device, dip=90.0), "dip")
    assert abs(at90 - near) < 0.2 * max(abs(near), 1e-12), \
        f"d/d(dip): {near!r} at dip=89.5 but {at90!r} at dip=90.0"


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
@pytest.mark.parametrize("dip", [89.999, 89.9999, 89.99999, 90.0, 90.00001])
def test_gradient_wrt_dip_is_smooth_through_vertical(okada, with_z, dip):
    """A-3 regression, the wider half.

    The inclined-fault formulae compute d/d(dip) as the difference of two terms
    that each diverge like 1/cos(dip).  Between |cos dip| ~ 1e-5 and 1e-4 that
    cancellation used to destroy the result (45% wrong at 89.999, 16600% and
    sign-flipped at 89.9999) -- a band roughly 100x wider than the snap band
    where the gradient was simply zero."""
    device = torch.device("cpu")
    coords = make_coords(device, with_z=with_z)
    reference = _ad_grad(okada, coords, make_params(device, dip=89.5), "dip")
    got = _ad_grad(okada, coords, make_params(device, dip=dip), "dip")
    assert abs(got - reference) < 0.1 * abs(reference), \
        f"d/d(dip) at dip={dip} is {got!r} but {reference!r} at 89.5"


def test_a_vertical_fault_can_be_optimised(okada):
    """The practical symptom of A-3: an inversion initialised at exactly dip=90
    could not move, because the gradient there was exactly zero."""
    device = torch.device("cpu")
    coords = make_coords(device, n=8)
    truth = make_params(device, dip=72.0)
    with torch.no_grad():
        obs = okada.compute(coords, truth, compute_strain=False)

    dip = torch.tensor(90.0, device=device, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([dip], lr=0.5)
    for _ in range(200):
        p = dict(truth); p["dip"] = dip
        u = okada.compute(coords, p, compute_strain=False)
        loss = sum(((a - b) ** 2).sum() for a, b in zip(u, obs))
        opt.zero_grad(); loss.backward(); opt.step()
    assert abs(dip.item() - 72.0) < 0.1, \
        f"starting from dip=90 the optimiser reached {dip.item()}, expected ~72"


def test_source_parameters_can_be_vmapped(okada):
    """A-4 regression.  Batching over source parameters is what makes multi-fault
    models and parallel MCMC chains cheap; it needs setup() to be branch-free."""
    from torch.func import vmap
    device = torch.device("cpu")
    coords = make_coords(device)

    def f(rake):
        p = dict(make_params(device))
        p["rake"] = rake
        return okada.compute(coords, p, compute_strain=False)[0]

    out = vmap(f)(torch.tensor([70.0, 80.0, 90.0]))
    assert out.shape[0] == 3
