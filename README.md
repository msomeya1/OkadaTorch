# OkadaTorch

`OkadaTorch` provides PyTorch implementations of FORTRAN subroutines that calculate displacements and strains (spatial derivative of displacements) due to a point source or a rectangular fault (Okada 1985, 1992).

**Features**
- **The whole code is differentiable**: the gradient with respect to the input can be easily computed using automatic differentiation (AD), allowing for flexible gradient-based optimization.
- **No for-loop over observation stations**: vectorization allows rapid calculation for multiple stations.
- **Easily combined with other models written in PyTorch**.



**References**
- Okada, Y. (1985). Surface deformation due to shear and tensile faults in a half-space. Bulletin of the seismological society of America, 75(4), 1135-1154.
https://doi.org/10.1785/BSSA0750041135
- Okada, Y. (1992). Internal deformation due to shear and tensile faults in a half-space. Bulletin of the seismological society of America, 82(2), 1018-1040.
https://doi.org/10.1785/BSSA0820021018
- [Program to calculate deformation due to a fault model DC3D0 / DC3D](https://www.bosai.go.jp/information/dc3d_e.html) (NIED website) 


Programs published in this repository are different from the original programs published in the NIED website.
The author have obtained permission from NIED to publish these programs here.




<!-- TODO:プレプリントへのリンク -->



## Install

Run
```shell
git clone https://github.com/msomeya1/OkadaTorch.git
cd OkadaTorch
pip install .
```



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

In addition, we provide convenient wrapper class, `OkadaWrapper`. 
Its usage can be found in [docs/OkadaWrapper.md](docs/OkadaWrapper.md).



If you find any bugs while using these programs, please let us know.

## Remark 1: Tensors

**In `OkadaTorch`, almost all variables must be treated as `torch.Tensor`.**
This is true for both the PyTorch implementation of original subroutines (`SPOINT`, `SRECTF`, `DC3D0`, `DC3D`) and the `OkadaWrapper` class.



Strictly speaking, the functions will work if some variables are just floats.
Variables must be passed as tensors in the following cases:
- Coordinate variables (`x,y(,z)`)
- Angle variables (`strike`, `dip`, and `rake`): when their sine and cosine are calculated internally, functions such as `torch.deg2rad`, `torch.sin`, `torch.cos` are used. These functions require that the argument be tensors.
- In case you want to differentiate the output by that variable. For example, if you want to differentiate by `depth`, you need to write `depth = torch.tensor(1.0, requires_grad=True)`. If you simply declare it as a float (e.g., `depth = 1.0`), the output is not differentiable with respect to `depth`.

We **believe** variables that are neither `x,y(,z)` nor angle variables, and whose derivatives are not calculated, can be declared as floats.
However, if you find it bothering to mix tensors and floats, it would be a good idea to declare all variables as tensors.



## Remark 2: Vectorization


Vectorization is performed only over stations and not over source parameters. 
This means
- displacements and strains at multiple stations can be obtained in batches [^1],
- but displacements and strains for multiple sources cannot be obtained in batches (only source parameters of a single source are acceptable).

This is due to technical reasons and we apologize for inconveniences.
If you want to compute displacements and strains for multiple sources, call the function multiple times.


[^1]: In this case, `x,y(,z)` will be 1D, 2D or 3D tensors with **same shape**. 
Returns (displacements and strains) are also tensors of the same shape.



## Remark 3: Coordinate System and Notation



The coordinate system used in functions `SPOINT`, `SRECTF`, `DC3D0` and `DC3D` is defined so that 
- the x-axis is parallel to the strike direction of the fault, 
- the z-axis is vertically upward, 
- and the y-axis is determined so that the entire system is right-handed.

However, `OkadaWrapper` uses a Cartesian coordinate system in which east is x, north is y, and up is z.




Also, the original FORTRAN subroutines and thier PyTorch implementations use uppercase variables (e.g., `UX`), while the OkadaWrapper uses lowercase variables (e.g., `ux`), but there is no particular difference between them (**except for the coordinate system difference noted above**). 
For example,
- `U1`, `UX`, and `ux` all represent the x component of the displacement.
- `U12`, `UXY`, and `uxy` all represent the x component of the displacement differentiated by y. 

> [!NOTE]
> `Uij` or `uij` means $\frac{\partial U_i}{\partial x_j}$ or $\frac{\partial u_i}{\partial x_j}$, respectively ($i,j=x,y,z$).
> In other words, the first index represents the component of displacement, and the second one represents which variable to differentiate.


## Hint

[`torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) is a useful feature that has the potential to speed up calculations by simply wrapping a function.
Let's consider the following code snippets as an example.
```python
okada = OkadaWrapper()
out = okada.compute(coords, params) 
```
If you rewrite it like this, 
```python
okada = OkadaWrapper()
compute_compiled = torch.compile(okada.compute) 
out = compute_compiled(coords, params) 
```
you can expect it to be faster.
 
However, as far as the author has tried, this seems to work only for `okada.compute`, and it had little effect on `okada.gradiet`, etc.