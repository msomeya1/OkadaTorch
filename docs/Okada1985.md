# How to use `SPOINT` and `SRECTF`

> [!NOTE]
> In the following, `Uij` means $\frac{\partial U_i}{\partial x_j}$.


## `SPOINT`

Calculate surface displacement, strain, tilt due to buried point source in a semiinfinite medium.


### Inputs

- `ALP` : _float or torch.Tensor_
    - Medium constant. $\frac{\mu}{\lambda+\mu}$
- `X, Y` : _torch.Tensor_
    - Coordinate of station.
- `D` : _float or torch.Tensor_
    - Source depth.
- `SD, CD` : _float or torch.Tensor_
    - Sine, cosine of dip-angle. 
    (CD=0.0, SD=+/-1.0 should be given for vertical fault.)
- `DISL1, DISL2, DISL3` : _float or torch.Tensor_
    - Strike-, dip- and tensile-dislocation.
- `compute_strain` : _bool, dafault True_
    - Option to calculate the spatial derivative of the displacement. 
    New in the PyTorch implementation.

### Outputs

If `compute_strain` is `True`, return is a list of 3 displacements and 6 spatial derivatives:
`[U1, U2, U3, U11, U12, U21, U22, U31, U32]`

If `False`, return is a list of 3 displacements only:
`[U1, U2, U3]`

- `U1, U2, U3` : _torch.Tensor_
    - Displacement. 
    unit = (unit of dislocation) / area
- `U11, U12, U21, U22` : _torch.Tensor_
    - Strain. 
    unit = (unit of dislocation) / (unit of X,Y,D) / area
- `U31, U32` : _torch.Tensor_
    - Tilt. 
    unit = (unit of dislocation) / (unit of X,Y,D) / area















<!-- ### Examples



```python
import numpy as np
import torch
from OkadaTorch import SPOINT, SRECTF

x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)
x, y = np.meshgrid(x, y)
x = torch.from_numpy(x)
y = torch.from_numpy(y)

dip = 45
sd = torch.sin(torch.deg2rad(dip))
cd = torch.cos(torch.deg2rad(dip))

disl1 = 1
disl2 = 1


``` -->


---

    
## `SRECTF`
Calculate surface displacements, strains and tilts due to rectangular fault in a half-space.

### Inputs

- `ALP` : _float or torch.Tensor_
    - Medium constant. $\frac{\mu}{\lambda+\mu}$
- `X, Y` : _torch.Tensor_
    - Coordinate of station.
- `D` : _float or torch.Tensor_
    - Source depth.
- `AL, AW` : _float or torch.Tensor_
    - Length and width of fault.
- `SD, CD` : _float or torch.Tensor_
    - Sin, Cosine of dip-angle. 
    (CD=0.0, SD=+/-1.0 should be given for vertical fault.)
- `DISL1, DISL2, DISL3` : _float or torch.Tensor_
    - Strike-, dip- and tensile-dislocation.
- `compute_strain` : _bool, dafault True_
    - Option to calculate the spatial derivative of the displacement. 
    New in the PyTorch implementation.

### Outputs

If `compute_strain` is `True`, return is a list of 3 displacements and 6 spatial derivatives:
`[U1, U2, U3, U11, U12, U21, U22, U31, U32]`

If `False`, return is a list of 3 displacements only:
`[U1, U2, U3]`

- `U1, U2, U3` : _torch.Tensor_
    - Displacement. unit = (unit of dislocation) 
- `U11, U12, U21, U22` : _torch.Tensor_
    - Strain. unit = (unit of dislocation) / (unit of X,Y, ... , AW)
- `U31, U32` : _torch.Tensor_
    - Tilt. unit = (unit of dislocation) / (unit of X,Y, ... , AW)




<!-- ### Examples



```python
import numpy as np
import torch
from OkadaTorch import SPOINT, SRECTF

x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)
x, y = np.meshgrid(x, y)
x = torch.from_numpy(x)
y = torch.from_numpy(y)

dip = 45
sd = torch.sin(torch.deg2rad(dip))
cd = torch.cos(torch.deg2rad(dip))

disl1 = 1
disl2 = 1

ux, uy, uz, uxx, uxy, uyx, uyy, uzx, uzy = SRECTF(0.5, X, Y, 0.1, 0.2, 0.2, sd, cd, disl1, disl2, 0, True)
``` -->