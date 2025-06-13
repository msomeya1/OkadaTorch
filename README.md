# OkadaTorch

`OkadaTorch` provides PyTorch implementations of FORTRAN subroutines that calculate displacements and strains due to a point source or a rectangular fault (Okada 1985, 1992).

**Features**
- **The whole code is differentiable**: the gradient with respect to the input can be easily computed using reverse-mode automatic differentiation (AD), allowing for flexible gradient-based optimization.
- **No for-loop over observation stations**: Vectorization allows instantaneous calculation for multiple stations.
- **Easily combined with other models written in PyTorch**.



**References**
- Okada, Y. (1985). Surface deformation due to shear and tensile faults in a half-space. Bulletin of the seismological society of America, 75(4), 1135-1154.
https://doi.org/10.1785/BSSA0750041135
- Okada, Y. (1992). Internal deformation due to shear and tensile faults in a half-space. Bulletin of the seismological society of America, 82(2), 1018-1040.
https://doi.org/10.1785/BSSA0820021018
- [Program to calculate deformation due to a fault model DC3D0 / DC3D](https://www.bosai.go.jp/information/dc3d_e.html). 



**TODO/pending**
- Easy-to-understand documents and examples (sorry current version is pretty bad)
- Exception handling (singular case and $z>0$ case). 
    - Return `IRET` as a tensor ? 
    - Return zero/Nan value in disp/strain ? 
    - Throw error and stop program ?
    - Others ?




# Install

To be written.

<!-- Run
```
git clone https://github.com/msomeya1/OkadaTorch.git
cd OkadaTorch
pip install .
```

`OkadaTorch` itself only requires PyTorch (which is installed in the steps above). 
However, if you want to run the example notebooks, you need additional packages (NumPy, Matplotlib, pyproj and Pyro). -->




# Usage



Okada (1985, 1992) provides four subroutines:
- `SPOINT` (Okada 1985): Calculate displacements and strains (spatial derivative of displacements) at surface ($z=0$) created by a point source.
- `SRECTF` (Okada 1985): Calculate displacements and strains at surface ($z=0$) created by a rectangular fault.
- `DC3D0` (Okada 1992): Same as `SPOINT`, but in depth ($z\leq0$).
- `DC3D` (Okada 1992): Same as `SRECTF`, but in depth ($z\leq0$).


||At Surface (Okada 1985)|In Depth (Okada 1992)|
|-|-|-|
|Point Source|`SPOINT`|`SRECTF`|
|Rectangular Fault|`DC3D0`|`DC3D`|


We have ported all of these subroutines to PyTorch.
<!-- Their usage can be found below.
- `SPOINT` and `SRECTF`: [docs/Okada1985.md](docs/Okada1985.md)
- `DC3D0` and `DC3D`: [docs/Okada1992.md](docs/Okada1992.md) -->

In addition, we provide convenient wrapper class, `OkadaWrapper`. 
<!-- Their usage can be found in [docs/OkadaWrapper.md](docs/OkadaWrapper.md). -->





