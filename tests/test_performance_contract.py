"""Structural properties that the performance of this library rests on.

None of these measures a wall-clock time. Timings on a shared machine are too
noisy to gate on, and the properties that actually matter here are structural:

* the whole of ``compute`` traces to **one** graph with no breaks, so
  ``torch.compile`` can fuse it;
* it performs **no host-device synchronisation**, so ``mode="reduce-overhead"``
  can capture it as a CUDA graph;
* it issues a bounded number of aten operations.

All three were hard-won. A single ``if some_tensor != 0:`` reintroduces a graph
break *and* a synchronisation, and both are invisible in ordinary use: the
answers stay correct, the code just quietly becomes an order of magnitude
slower. See DEBUG_NOTES.md sections 5 and A-1.

The first and third run on CPU, so CI covers them. The synchronisation test
needs a GPU and is skipped elsewhere.
"""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from conftest import make_coords, make_params


class _CountAtenOps(TorchDispatchMode):
    def __init__(self):
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += 1
        return func(*args, **(kwargs or {}))


def _case(name, device):
    """The six shapes of call, by (surface|depth) x (rect|point) x (disp|full)."""
    surface, rect, strain = {
        "surface_rect_disp":   (True,  True,  False),
        "surface_rect_full":   (True,  True,  True),
        "depth_rect_disp":     (False, True,  False),
        "depth_rect_full":     (False, True,  True),
        "surface_point_disp":  (True,  False, False),
        "depth_point_disp":    (False, False, False),
    }[name]
    coords = make_coords(device, with_z=not surface, n=25)
    return coords, make_params(device, rect=rect), strain


# Measured on the current implementation, rounded up by ~25%.  The budget is a
# guard against a gross regression -- a loop that stopped being vectorised, a
# block of work that is no longer skipped -- not a target to optimise against.
#
# If a deliberate change pushes a number past its budget, re-measure and raise
# the budget in the same commit, so the new cost is recorded rather than hidden.
ATEN_OP_BUDGET = {
    "surface_rect_disp":   1718,
    "surface_rect_full":   3653,
    "depth_rect_disp":     4321,
    "depth_rect_full":    11310,
    "surface_point_disp":   218,
    "depth_point_disp":     908,
}


@pytest.mark.parametrize("name", sorted(ATEN_OP_BUDGET))
def test_aten_operation_count_stays_within_budget(okada, name):
    coords, params, strain = _case(name, torch.device("cpu"))
    counter = _CountAtenOps()
    with counter:
        okada.compute(coords, params, compute_strain=strain)
    budget = ATEN_OP_BUDGET[name]
    assert counter.count <= budget, (
        f"{name} now issues {counter.count} aten operations, over its budget of "
        f"{budget}. If the increase is intended, re-measure and update "
        f"ATEN_OP_BUDGET in this file.")


@pytest.mark.parametrize("name", ["surface_rect_disp", "depth_rect_full",
                                  "surface_point_disp"])
def test_compute_traces_to_a_single_graph(okada, name):
    """No graph breaks.

    Every break used to come from a Python `if` on a tensor value -- 26
    `if DISLn != 0.0` guards and two `if CD != 0.0` branches.  Those also
    dropped the derivative of whatever they skipped (A-1), so this test guards
    correctness as much as speed.

    `torch._dynamo.explain` is a private API; if a torch upgrade moves it, fix
    the test rather than deleting it.
    """
    import torch._dynamo as dynamo

    coords, params, strain = _case(name, torch.device("cpu"))
    dynamo.reset()
    explanation = dynamo.explain(
        lambda: okada.compute(coords, params, compute_strain=strain))()
    reasons = "\n".join(f"  - {r.reason}" for r in explanation.break_reasons)
    assert explanation.graph_break_count == 0, (
        f"{name} now has {explanation.graph_break_count} graph break(s):\n{reasons}")
    assert explanation.graph_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("name", sorted(ATEN_OP_BUDGET))
def test_compute_never_synchronises_with_the_host(okada, name):
    """A Python `if` on a CUDA tensor blocks until the value reaches the host.

    There used to be 13 to 41 of these per call, and they were the main reason a
    GPU run was *slower* than a CPU one.  A graph containing one also cannot be
    captured by `torch.compile(mode="reduce-overhead")`.

    `set_sync_debug_mode("error")` turns any synchronisation into a RuntimeError.
    """
    index = 1 if torch.cuda.device_count() > 1 else 0
    device = torch.device(f"cuda:{index}")
    coords, params, strain = _case(name, device)

    okada.compute(coords, params, compute_strain=strain)   # warm up allocators
    torch.cuda.synchronize(device)
    torch.cuda.set_sync_debug_mode("error")
    try:
        okada.compute(coords, params, compute_strain=strain)
    except RuntimeError as exc:
        pytest.fail(f"{name} synchronises with the host: {exc}")
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_backward_never_synchronises_with_the_host(okada):
    """The same requirement for the backward pass, which is what an optimiser
    or an MCMC sampler actually spends its time in."""
    index = 1 if torch.cuda.device_count() > 1 else 0
    device = torch.device(f"cuda:{index}")
    coords = make_coords(device, n=25)
    params = {k: v.clone().requires_grad_(True)
              for k, v in make_params(device).items()}

    def step():
        for v in params.values():
            v.grad = None
        ux, uy, uz = okada.compute(coords, params, compute_strain=False)
        (ux.square().sum() + uy.square().sum() + uz.square().sum()).backward()

    step()
    torch.cuda.synchronize(device)
    torch.cuda.set_sync_debug_mode("error")
    try:
        step()
    except RuntimeError as exc:
        pytest.fail(f"the backward pass synchronises with the host: {exc}")
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_source_parameters_stay_vmappable(okada):
    """Batching over sources is what makes a multi-fault model cheap.

    It works only because no branch depends on a parameter value; the first
    `if` on one would break it again, and the failure would look like an
    unrelated vmap error rather than a performance problem.
    """
    from torch.func import vmap

    device = torch.device("cpu")
    coords = make_coords(device, n=6)
    base = make_params(device)
    keys = sorted(base)

    def one_fault(values):
        params = dict(zip(keys, values))
        return torch.stack(okada.compute(coords, params, compute_strain=False))

    faults = torch.stack([torch.stack([base[k] * s for k in keys])
                          for s in (0.9, 1.0, 1.1)])
    out = vmap(one_fault)(faults)
    assert out.shape == (3, 3, *coords["x"].shape)
    assert torch.isfinite(out).all()
