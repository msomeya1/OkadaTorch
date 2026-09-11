# OkadaTorch

[![tests](https://github.com/msomeya1/OkadaTorch/actions/workflows/test.yml/badge.svg)](https://github.com/msomeya1/OkadaTorch/actions/workflows/test.yml)

`OkadaTorch` provides PyTorch implementations of FORTRAN subroutines that calculate displacements and strains (spatial derivatives of displacements) due to a point source or a rectangular fault (Okada 1985, 1992).

**Features**
- **The whole code is differentiable**: the gradient with respect to the input can be easily computed using automatic differentiation (AD), allowing for flexible gradient-based optimization.
- **No for-loop over observation stations and sources**: vectorization allows rapid calculation for multiple stations and sources.
- **Easily combined with other models written in PyTorch**.



**References**
- Okada, Y. (1985). Surface deformation due to shear and tensile faults in a half-space. Bulletin of the seismological society of America, 75(4), 1135-1154.
https://doi.org/10.1785/BSSA0750041135
- Okada, Y. (1992). Internal deformation due to shear and tensile faults in a half-space. Bulletin of the seismological society of America, 82(2), 1018-1040.
https://doi.org/10.1785/BSSA0820021018
- [Program to calculate deformation due to a fault model DC3D0 / DC3D](https://www.bosai.go.jp/information/dc3d_e.html) (NIED website) 


Programs published in this repository are different from the original programs published on the NIED website.
The author has obtained permission from NIED to publish these programs here.




If you use `OkadaTorch` in your study, please consider citing the following preprint.
- Masayoshi Someya, Taisuke Yamada, Tomohisa Okazaki. OkadaTorch: A Differentiable Programming of Okada Model to Calculate Displacements and Strains from Fault Parameters, arXiv preprint (2025). https://arxiv.org/abs/2507.17126


If you find any bugs when using `OkadaTorch`, please let us know.


## Install

Execute:
```shell
git clone https://github.com/msomeya1/OkadaTorch.git
cd OkadaTorch
pip install .
```

To run the example notebooks as well, execute:
```shell
pip install ".[examples]"
```

Confirmed to work with PyTorch 2.5.1, 2.6.0 (CPU) and 2.7.1 (CPU and CUDA 12).



## Usage

Okada (1985, 1992) provides four subroutines:
- `SPOINT` (Okada 1985): Calculate displacements and strains at the surface ($z=0$) created by a point source.
- `SRECTF` (Okada 1985): Calculate displacements and strains at the surface ($z=0$) created by a rectangular fault.
- `DC3D0` (Okada 1992): Same as `SPOINT`, but under the surface ($z\leq0$).
- `DC3D` (Okada 1992): Same as `SRECTF`, but under the surface ($z\leq0$).


||At Surface (Okada 1985)|Under Surface (Okada 1992)|
|-|-|-|
|Point Source|`SPOINT`|`DC3D0`|
|Rectangular Fault|`SRECTF`|`DC3D`|


We have ported all of these subroutines into PyTorch.
Their usage can be found in the following.
- `SPOINT` and `SRECTF`: [docs/Okada1985.md](docs/Okada1985.md)
- `DC3D0` and `DC3D`: [docs/Okada1992.md](docs/Okada1992.md)

In addition, we provide a convenient wrapper class, `OkadaWrapper`. 
Its usage can be found in [docs/OkadaWrapper.md](docs/OkadaWrapper.md).





## Remark 1: Tensors

`OkadaWrapper` accepts plain Python numbers as well as tensors; both `strike = 189.0` and `strike = torch.tensor(189.0)` are okay.

Two things still need a tensor:

- **Anything you want to differentiate.** `depth = torch.tensor(1.0, requires_grad=True)` is differentiable; `depth = 1.0` is a constant.
- **The station coordinates**, if you want more than a single point.


## Remark 2: Vectorization

Stations are vectorized directly: pass `x,y(,z)` as tensors of any shape (they must all have the same shape), 
and the returned displacements and strains have the same shape.

Sources are not vectorized directly: every source parameter must be a scalar. 
Multiple sources are handled by `torch.func.vmap`, and since the Okada solution 
is linear with respect to source, summation over the batch dimension provides 
the multi-source solution.
See [Multiple sources](docs/OkadaWrapper.md#multiple-sources) for example code.


## Remark 3: Coordinate System and Notation



The coordinate system used in functions `SPOINT`, `SRECTF`, `DC3D0` and `DC3D` is defined so that 
- the x-axis is parallel to the strike direction of the fault, 
- the z-axis is vertically upward, 
- and the y-axis is determined so that the entire system is right-handed.

However, `OkadaWrapper` uses a Cartesian coordinate system in which east is x, north is y, and up is z.




Also, the original FORTRAN subroutines and their PyTorch implementations use uppercase variables (e.g., `UX`), while the OkadaWrapper uses lowercase variables (e.g., `ux`), but there is no particular difference between them (**except for the coordinate system difference noted above**). 
For example,
- `U1`, `UX`, and `ux` all represent the x component of the displacement.
- `U12`, `UXY`, and `uxy` all represent the x component of the displacement differentiated by y. 

> [!NOTE]
> `Uij` or `uij` means $\frac{\partial U_i}{\partial x_j}$ or $\frac{\partial u_i}{\partial x_j}$, respectively ($i,j=x,y,z$).
> In other words, the first index represents the component of displacement, and the second one represents which variable to differentiate.


## Remark 4: Precision

`float32` is the default dtype in `PyTorch` and a reasonable choice when using `OkadaTorch` alongside neural network models. However, output and gradient accuracy are lower than with `float64`, and the problem is more serious for the gradients.
Therefore, `float64` is recommended when precise gradients are required (e.g., gradient-based optimization) or when dealing with shallow faults (see below). A warning is displayed if `float32` or lower precision is used.


### Faults that reach the surface

When `depth = 0` (`fault_origin="topleft"`), i.e., the upper edge of the fault reaches the surface, the displacement at points on the fault trace is finite, but the gradient is not (typically `NaN`). To avoid issues with gradients, we set the output at such points to exactly zero and report them with `IRET = 1` (see [Return codes](docs/OkadaWrapper.md#return-codes)).

In practice, this issue occurs within a finite-width zone around the trace. When computing displacements on a dense grid (e.g., seafloor displacement for tsunami simulations), some grid points may fall within this zone. The width of the zone depends on machine epsilon (and hence on the dtype). For `float64`, the width is on the order of nanometers, so this problem is practically negligible. For `float32`, however, the width is on the order of meters, meaning that some grid points may return zero displacement despite the true value being finite. Therefore, **`float64` is recommended for faults that may reach the surface.**



## Remark 5: Performance Hint

[`torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) is a technique for accelerating computation by compiling models written in `PyTorch`. For example, you can speed up computation simply by writing [^1]:

```python
okada = OkadaWrapper()
compute_compiled = torch.compile(okada.compute)
out = compute_compiled(coords, params)
```

[^1]: In v0.1.0, the documentation stated that the effect of `torch.compile` was limited. This was because the code was not written in a way that avoided graph breaks, preventing `torch.compile` from being fully effective. Since v0.2.0, the code has been rewritten to eliminate graph breaks, allowing `torch.compile` to deliver its full performance benefit.

If the tensor shapes do not change across iterations, enabling `CUDA graphs` can provide further speedup:

```python
compute_compiled = torch.compile(okada.compute, mode="reduce-overhead")
```

Note that in this mode, the returned tensors are backed by static buffers that are reused across calls. Therefore, you must use `.clone()` for any data that needs to be retained across iterations.


## License

[MIT LICENSE](LICENSE). 

This covers the PyTorch implementation in this repository and not the original NIED programs.

## Version history

### 0.2.0

Debugging and refactoring.

- **Gradient issues** 
  - Removed `if DISLn != 0.0` (n=1,2,3). In the previous code, which closely followed the original FORTRAN implementation, gradients from the skipped terms inside these conditional branches were not computed. This meant that incorrect gradients were returned when `rake=0` or `slip=0`. In the new code, the `if DISLn != 0.0` is simply removed (adding zero terms does not affect the output while correctly computing the gradient).
  - Fixed `NaN` gradients caused by `torch.where`. When unselected branches of `torch.where(condition, a, b)` contained `inf` or similar values, `NaN` could propagate into the computed gradients. This was resolved by applying safe-guarding operations (e.g., avoiding division by zero) before passing values to `torch.where`.
  - Fixed incorrect gradients `d/d(dip)` for vertical faults (`dip = 90`). Instead of directly differentiating the special-case formula for vertical faults, we now pass a surrogate gradient derived from the inclined-fault formula evaluated at `dip ≈ 90`. This ensures that both the outputs and gradients are correct.
  - Replaced conditional statements involving differentiable parameters with `torch.where` to prevent graph breaks. This allows `torch.compile` to be fully effective and also enables batched computation over multiple faults using `vmap`.

- **Singular station treatment** 
  - At singular stations, the output is exactly zero, and
  `compute(..., return_iret=True)` reports which ones they are. Simply masking the output would leave the gradients as `NaN`, so dummy coordinates are assigned to the singular points to allow the computation to proceed, and zeros are substituted at the end.

- **New features**
  - `params["opening"]` (tensile) and `params["inflation"]` (isotropic; point source with `z` only, since only `DC3D0` has that component) are now supported in `OkadaWrapper`.
  - `torch.func.vmap` over every source parameter is supported (multiple faults can be evaluated easily).
  - Plain Python numbers are accepted for any parameter.
- **Input validation** 
  - Raises `ValueError` instead of `AssertionError`
  - Unrecognized keys are rejected rather than ignored.
  - Mixed devices are rejected.

- **Units** 
  - Tolerances are relative, so the same fault gives the same answer
  whether you work in meters or kilometers.
- **Performance**
  - `torch.compile` can now trace the whole kernel into a single graph and the kernel no longer synchronizes with the host.
- **Added tests** 


### 0.1.0

Initial release.
