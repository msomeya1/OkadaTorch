"""The public contract of OkadaWrapper: shapes, dtypes, devices, validation.

Most of these are cheap guards that would have caught the C-* and D-* issues in
DEBUG_NOTES.md.  The ones that currently fail are marked ``xfail(strict=True)``.
"""
import numpy as np
import pytest
import torch

from conftest import make_coords, make_params


# ---------------------------------------------------------------------------
# shapes and containers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(7,), (3, 4), (2, 3, 4)])
@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_output_shape_follows_coordinate_shape(okada, device, shape, with_z):
    x = torch.rand(shape, device=device, dtype=torch.float64) * 100 - 50
    y = torch.rand(shape, device=device, dtype=torch.float64) * 100 - 50
    coords = {"x": x, "y": y}
    if with_z:
        coords["z"] = -torch.rand(shape, device=device, dtype=torch.float64) * 10 - 1
    for strain, n in ((False, 3), (True, 12)):
        out = okada.compute(coords, make_params(device), compute_strain=strain)
        assert len(out) == n
        assert all(t.shape == shape for t in out)


def test_gradient_and_hessian_preserve_coordinate_shape(okada, device):
    coords = make_coords(device, with_z=True, n=4)
    params = make_params(device)
    for g in okada.gradient(coords, params, arg="x", compute_strain=False):
        assert g.shape == coords["x"].shape
    for h in okada.hessian(coords, params, arg1="x", arg2="y", compute_strain=False):
        assert h.shape == coords["x"].shape


@pytest.mark.parametrize("where,key", [("coords", "Z"), ("coords", "unused"),
                                       ("params", "widht"), ("params", "openning"),
                                       ("params", "dpeth")])
def test_unknown_keys_raise(okada, device, where, key):
    """Unknown keys used to be ignored, which is exactly what made a typo
    dangerous: 'Z' silently selected the surface formulation, 'widht' silently
    selected a point source, 'openning' silently dropped the tensile term."""
    coords, params = make_coords(device), make_params(device)
    target = coords if where == "coords" else params
    target[key] = torch.tensor(1.0, device=device, dtype=torch.float64)
    with pytest.raises(ValueError, match="unrecognized"):
        okada.compute(coords, params, compute_strain=False)


def test_inputs_are_not_mutated(okada, device):
    coords, params = make_coords(device), make_params(device)
    before_c = {k: v.clone() for k, v in coords.items()}
    before_p = {k: v.clone() for k, v in params.items()}
    okada.compute(coords, params, compute_strain=True)
    for k in coords:
        assert torch.equal(coords[k], before_c[k]), f"coords['{k}'] was mutated"
    for k in params:
        assert torch.equal(params[k], before_p[k]), f"params['{k}'] was mutated"


# ---------------------------------------------------------------------------
# device / dtype
# ---------------------------------------------------------------------------
def test_output_stays_on_the_input_device(okada, device):
    out = okada.compute(make_coords(device, with_z=True), make_params(device),
                        compute_strain=True)
    assert all(t.device == device for t in out)


def test_uniform_dtype_is_preserved(okada, device):
    coords = make_coords(device, dtype=torch.float64, with_z=True)
    params = make_params(device, dtype=torch.float64)
    out = okada.compute(coords, params, compute_strain=True)
    assert all(t.dtype == torch.float64 for t in out)


def test_float32_works_but_warns(okada, device):
    """D-2.  float32 is not rejected -- around 0.2% on the displacements, which
    is fine next to a neural network -- but it is a choice worth making
    knowingly, so it must not happen silently."""
    coords = make_coords(device, dtype=torch.float32, with_z=True)
    params = make_params(device, dtype=torch.float32)
    with pytest.warns(UserWarning, match="float32"):
        out = okada.compute(coords, params, compute_strain=True)
    assert all(t.dtype == torch.float32 for t in out)


# ---------------------------------------------------------------------------
# validation -- these describe the contract we *want*
# ---------------------------------------------------------------------------
def test_missing_required_keys_raise(okada, device):
    """ValueError rather than AssertionError: `python -O` strips assertions, and
    validation that disappears under -O is validation you cannot rely on."""
    coords, params = make_coords(device), make_params(device)
    with pytest.raises(ValueError, match="missing"):
        okada.compute({"x": coords["x"]}, params, compute_strain=False)
    with pytest.raises(ValueError, match="missing"):
        okada.compute(coords, {k: v for k, v in params.items() if k != "dip"},
                      compute_strain=False)


@pytest.mark.parametrize("method", ["compute", "gradient", "hessian"])
def test_all_three_methods_validate_identically(okada, device, method):
    """The three entry points used to repeat the same checks by hand, and had
    drifted apart; they now share one validator."""
    params = {k: v for k, v in make_params(device).items() if k != "slip"}
    kw = {"gradient": dict(arg="depth"), "hessian": dict(arg1="depth", arg2="dip")}
    with pytest.raises(ValueError, match="missing"):
        getattr(okada, method)(make_coords(device), params, compute_strain=False,
                               **kw.get(method, {}))


def test_mismatched_coordinate_shapes_raise(okada, device):
    coords = make_coords(device)
    coords["y"] = coords["y"][:-1]
    with pytest.raises(ValueError, match="same shape"):
        okada.compute(coords, make_params(device), compute_strain=False)


def test_invalid_gradient_arg_raises(okada, device):
    with pytest.raises(ValueError):
        okada.gradient(make_coords(device), make_params(device),
                       arg="not_a_parameter", compute_strain=False)


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
@pytest.mark.parametrize("origin", ["TopLeft", "centre", "", "top-left"])
def test_invalid_fault_origin_raises_valueerror(okada, device, with_z, origin):
    """C-1 regression.  The DC3D branch raised ValueError but the SRECTF branch
    fell through to `return out` with `out` unbound, so the same typo produced a
    ValueError or an UnboundLocalError depending on whether `z` was given."""
    coords = make_coords(device, with_z=with_z)
    with pytest.raises(ValueError, match="fault_origin"):
        okada.compute(coords, make_params(device), compute_strain=False,
                      fault_origin=origin)


def test_invalid_fault_origin_is_caught_even_for_a_point_source(okada, device):
    """`fault_origin` is ignored for a point source, but a typo in it is still a
    typo -- silently accepting it would hide the mistake until the model is
    later changed to a rectangle."""
    with pytest.raises(ValueError, match="fault_origin"):
        okada.compute(make_coords(device), make_params(device, rect=False),
                      compute_strain=False, fault_origin="TopLeft")


@pytest.mark.parametrize("drop", ["length", "width"])
def test_partial_rectangle_parameters_raise(okada, device, drop):
    """C-2 regression.  Dropping one of length/width used to fall through to the
    point-source branch without a word, so a typo such as 'widht' silently
    changed which physical model was being fitted."""
    params = {k: v for k, v in make_params(device).items() if k != drop}
    with pytest.raises(ValueError, match="rectangular fault"):
        okada.compute(make_coords(device), params, compute_strain=False)


def test_a_point_source_still_needs_neither_length_nor_width(okada, device):
    """The other half of C-2: dropping *both* is the legitimate way to ask for a
    point source and must keep working."""
    out = okada.compute(make_coords(device), make_params(device, rect=False),
                        compute_strain=False)
    assert len(out) == 3 and all(torch.isfinite(t).all() for t in out)


def test_hessian_reports_the_actual_problem(okada, device):
    """C-3.  An arg that is simply absent used to be rejected with the
    "both must be of the same kind" message, pointing at the wrong mistake."""
    coords, params = make_coords(device), make_params(device)   # no "z"
    with pytest.raises(ValueError, match="not a key"):
        okada.hessian(coords, params, arg1="x", arg2="z", compute_strain=False)
    with pytest.raises(ValueError, match="both be"):
        okada.hessian(coords, params, arg1="x", arg2="depth", compute_strain=False)


@pytest.mark.parametrize("nu", [0.5, -1.0, 1.5, 1.0, 100.0])
def test_invalid_poisson_ratio_raises(okada, device, nu):
    """C-4 regression.  nu=0.5 used to return all zeros (1-2nu = 0) and nu=1.5
    a plausible-looking but meaningless field, both without a word."""
    with pytest.raises(ValueError, match="Poisson"):
        okada.compute(make_coords(device), make_params(device),
                      compute_strain=True, nu=nu)


@pytest.mark.parametrize("nu", [0.0, 0.25, 0.3, 0.49])
def test_valid_poisson_ratios_are_accepted(okada, device, nu):
    out = okada.compute(make_coords(device), make_params(device),
                        compute_strain=True, nu=nu)
    assert all(torch.isfinite(t).all() for t in out)


def test_positive_z_is_flagged_rather_than_rejected(okada, device):
    """Deliberate: z > 0 is reported through IRET, not by raising.

    Raising would need `bool((z > 0).any())`, a host synchronisation, and a
    graph containing one cannot be captured by torch.compile's CUDA-graph mode
    -- which is where the 20x speed-up in DEBUG_NOTES.md section 5 comes from.
    The station is zeroed and flagged instead, exactly as the original FORTRAN
    does, and `return_iret=True` makes it visible."""
    coords = make_coords(device, with_z=True)
    coords["z"] = torch.abs(coords["z"])
    out, iret = okada.compute(coords, make_params(device), compute_strain=False,
                              return_iret=True)
    assert torch.all(iret == 2)
    assert all(torch.all(t == 0.0) for t in out)


@pytest.mark.parametrize("coord_dtype,param_dtype",
                         [(torch.float32, torch.float64),
                          (torch.float64, torch.float32)])
def test_mixed_dtypes_are_promoted(okada, device, coord_dtype, param_dtype):
    """D-2 regression.  `coords` used to win outright, so float64 parameters were
    quietly demoted.  Promotion fixes that without rejecting the mixture, which
    is what `torch.from_numpy` plus `torch.tensor(30.0)` naturally produces."""
    coords = make_coords(device, dtype=coord_dtype, with_z=True)
    params = make_params(device, dtype=param_dtype)
    out = okada.compute(coords, params, compute_strain=True)
    assert all(t.dtype == torch.float64 for t in out)


def test_promotion_gives_the_float64_answer(okada, device):
    """Promoting must produce the float64 answer for the values supplied, not
    merely a float64-typed container holding a float32 computation."""
    coords = make_coords(device, dtype=torch.float64, with_z=True)
    p32 = make_params(device, dtype=torch.float32)
    mixed = okada.compute(coords, p32, compute_strain=True)
    ref = okada.compute(coords, {k: v.double() for k, v in p32.items()},
                        compute_strain=True)
    assert all(torch.equal(a, b) for a, b in zip(mixed, ref))


def test_promotion_keeps_the_gradient(okada, device):
    """The cast to the promoted dtype has to stay inside the autograd graph."""
    coords = make_coords(device, dtype=torch.float64)
    params = make_params(device, dtype=torch.float32)
    params["slip"] = params["slip"].clone().requires_grad_(True)
    ux, _, _ = okada.compute(coords, params, compute_strain=False)
    ux.sum().backward()
    grad = params["slip"].grad
    assert grad is not None and grad.dtype == torch.float32
    assert torch.isfinite(grad).all() and grad != 0.0


def test_mixed_devices_are_rejected(okada):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    dev = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    with pytest.raises(ValueError, match="one device"):
        okada.compute(make_coords(dev), make_params(torch.device("cpu")),
                      compute_strain=False)


def test_scalar_python_floats_are_accepted(okada, device):
    """C-6 regression.  `strike=30.0` used to fail with a TypeError from inside
    torch.deg2rad, several frames below the user's call."""
    coords = make_coords(device)
    params = {k: float(v) for k, v in make_params(device).items()}
    out = okada.compute(coords, params, compute_strain=False)
    assert all(t.device == device and t.dtype == coords["x"].dtype for t in out)


def test_python_floats_give_the_same_answer_as_tensors(okada, device):
    coords = make_coords(device, with_z=True)
    tensors = make_params(device)
    floats = {k: float(v) for k, v in tensors.items()}
    a = okada.compute(coords, tensors, compute_strain=True)
    b = okada.compute(coords, floats, compute_strain=True)
    for u, v in zip(a, b):
        assert torch.equal(u, v)


@pytest.mark.parametrize("method", ["compute", "gradient", "hessian"])
def test_all_three_methods_accept_python_floats(okada, device, method):
    params = {k: float(v) for k, v in make_params(device).items()}
    kw = {"gradient": dict(arg="x"), "hessian": dict(arg1="x", arg2="y")}
    out = getattr(okada, method)(make_coords(device), params, compute_strain=False,
                                 **kw.get(method, {}))
    assert all(torch.isfinite(t).all() for t in out)


def test_scalar_coordinates_are_accepted(okada, device):
    """A single station given as plain numbers, the smallest sensible call."""
    params = make_params(device)
    out = okada.compute({"x": 12.0, "y": -34.0}, params, compute_strain=False)
    assert all(t.ndim == 0 for t in out)


# ---------------------------------------------------------------------------
# C-5: the tensile and isotropic source components
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
@pytest.mark.parametrize("rect", [False, True], ids=["point", "rect"])
def test_opening_is_accepted_and_changes_the_field(okada, device, with_z, rect):
    coords = make_coords(device, with_z=with_z)
    params = make_params(device, rect=rect)
    base = okada.compute(coords, params, compute_strain=True)
    with_open = okada.compute(coords, dict(params, opening=torch.tensor(
        2.0, device=device, dtype=torch.float64)), compute_strain=True)
    assert all(torch.isfinite(t).all() for t in with_open)
    assert not torch.allclose(torch.stack(base), torch.stack(with_open))


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
@pytest.mark.parametrize("rect", [False, True], ids=["point", "rect"])
def test_opening_defaults_to_zero(okada, device, with_z, rect):
    """Backwards compatibility: omitting `opening` must reproduce the old answer."""
    coords = make_coords(device, with_z=with_z)
    params = make_params(device, rect=rect)
    a = okada.compute(coords, params, compute_strain=True)
    b = okada.compute(coords, dict(params, opening=torch.zeros(
        (), device=device, dtype=torch.float64)), compute_strain=True)
    for u, v in zip(a, b):
        assert torch.equal(u, v)


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_source_components_superpose(okada, device, with_z):
    """The Okada solution is linear in the source, so a mixed source must equal
    the sum of its parts.  This is what makes `opening` safe to add: it cannot
    interact with `slip`/`rake`."""
    coords = make_coords(device, with_z=with_z)
    Z = lambda: torch.zeros((), device=device, dtype=torch.float64)
    shear = make_params(device, slip=3.0, rake=70.0)
    mixed = okada.compute(coords, dict(shear, opening=torch.tensor(
        2.0, device=device, dtype=torch.float64)), compute_strain=True)
    only_shear = okada.compute(coords, dict(shear, opening=Z()), compute_strain=True)
    only_open = okada.compute(coords, dict(shear, slip=Z(), opening=torch.tensor(
        2.0, device=device, dtype=torch.float64)), compute_strain=True)
    for m, a, b in zip(mixed, only_shear, only_open):
        assert torch.allclose(m, a + b, rtol=1e-10, atol=1e-14)


def test_opening_is_differentiable(okada, device):
    coords = make_coords(device, with_z=True)
    params = make_params(device)
    params["opening"] = torch.tensor(1.5, device=device, dtype=torch.float64,
                                     requires_grad=True)
    ux, uy, uz = okada.compute(coords, params, compute_strain=False)
    g = torch.autograd.grad((ux + uy + uz).sum(), params["opening"])[0]
    assert torch.isfinite(g).all() and g.item() != 0.0


def test_opening_agrees_between_the_1985_and_1992_paths(okada, device):
    coords = make_coords(device)
    params = make_params(device, rect=True)
    params["opening"] = torch.tensor(2.0, device=device, dtype=torch.float64)
    a = okada.compute(coords, params, compute_strain=True)
    b = okada.compute(dict(coords, z=torch.zeros_like(coords["x"])), params,
                      compute_strain=True)
    for u, v in zip(a, b):
        assert torch.allclose(u, v, rtol=1e-9, atol=1e-12)


def test_inflation_works_for_a_point_source_at_depth(okada, device):
    coords = make_coords(device, with_z=True)
    params = make_params(device, rect=False)
    params["inflation"] = torch.tensor(1.0, device=device, dtype=torch.float64,
                                       requires_grad=True)
    out = okada.compute(coords, params, compute_strain=True)
    assert all(torch.isfinite(t).all() for t in out)
    g = torch.autograd.grad(torch.stack(out).sum(), params["inflation"])[0]
    assert g.item() != 0.0


def test_inflation_is_rejected_where_no_kernel_supports_it(okada, device):
    """DC3D (rectangular) has no isotropic term, and SPOINT (surface point
    source) has only three components -- only DC3D0 has all four."""
    infl = torch.tensor(1.0, device=device, dtype=torch.float64)
    with pytest.raises(ValueError, match="point source"):
        okada.compute(make_coords(device, with_z=True),
                      dict(make_params(device, rect=True), inflation=infl),
                      compute_strain=False)
    with pytest.raises(ValueError, match="requires 'z'"):
        okada.compute(make_coords(device),
                      dict(make_params(device, rect=False), inflation=infl),
                      compute_strain=False)
