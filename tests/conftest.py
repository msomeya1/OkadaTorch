import os

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(HERE, "golden")

# The reference data is float64 throughout; running the suite in any other
# default dtype would silently weaken every tolerance in it.
torch.set_default_dtype(torch.float64)


def _devices():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        # honour CUDA_VISIBLE_DEVICES; index 1 is the GPU reserved for this work
        idx = 1 if torch.cuda.device_count() > 1 else 0
        devs.append(torch.device(f"cuda:{idx}"))
    return devs


@pytest.fixture(scope="session", params=_devices(), ids=lambda d: str(d))
def device(request):
    return request.param


@pytest.fixture(scope="session")
def fortran_golden():
    path = os.path.join(GOLDEN_DIR, "fortran_reference.npz")
    if not os.path.exists(path):
        pytest.skip("run tests/generate_golden.py first")
    return np.load(path)


@pytest.fixture(scope="session")
def wrapper_golden():
    path = os.path.join(GOLDEN_DIR, "wrapper_snapshot.npz")
    if not os.path.exists(path):
        pytest.skip("run tests/generate_golden.py first")
    return np.load(path)


@pytest.fixture
def okada():
    from OkadaTorch import OkadaWrapper
    return OkadaWrapper()


# ---------------------------------------------------------------- helpers
def rel_err(got, ref):
    """Element-wise error normalised by max(|ref|, 1).

    Plain relative error is useless here because many components are legitimately
    tiny; plain absolute error is useless because others are O(1e3).  This mixed
    measure is what the FORTRAN comparison in DEBUG_NOTES.md used.
    """
    got, ref = np.asarray(got, dtype=float), np.asarray(ref, dtype=float)
    return np.abs(got - ref) / np.maximum(np.abs(ref), 1.0)


def make_coords(device, dtype=torch.float64, with_z=False, n=5):
    """A small, well-conditioned station grid (km)."""
    x = torch.linspace(-180.0, 220.0, n, device=device, dtype=dtype)
    y = torch.linspace(-160.0, 210.0, n, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    coords = {"x": X.contiguous(), "y": Y.contiguous()}
    if with_z:
        coords["z"] = torch.full_like(X, -7.0)
    return coords


def make_params(device, dtype=torch.float64, rect=True, requires_grad=False, **over):
    base = dict(x_fault=3.0, y_fault=-11.0, depth=6.5, strike=189.0,
                dip=57.0, rake=101.0, slip=5.62)
    if rect:
        base.update(length=218.0, width=46.0)
    base.update(over)
    return {
        k: torch.tensor(v, device=device, dtype=dtype, requires_grad=requires_grad)
        for k, v in base.items()
    }


PARAM_NAMES = ["x_fault", "y_fault", "depth", "length", "width",
               "strike", "dip", "rake", "slip"]
