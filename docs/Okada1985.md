# How to use `SPOINT` and `SRECTF`



## `SPOINT`

Calculate surface displacement, strain, tilt due to buried point source in a semiinfinite medium.


### Inputs

- `ALP` : _float or torch.Tensor_
    - Medium constant, equal to $\frac{\mu}{\lambda+\mu}$.
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

If `compute_strain` is `True`, return is a list of 3 displacements and 6 spatial derivatives: \
`[U1, U2, U3, U11, U12, U21, U22, U31, U32]`

If `False`, return is a list of 3 displacements only:
`[U1, U2, U3]`

- `U1, U2, U3` : _torch.Tensor_
    - Displacement. 
    $\text{unit} = \frac{\text{(unit of dislocation)}}{\text{(area)}}$
- `U11, U12, U21, U22` : _torch.Tensor_
    - Strain. 
    $\text{unit} = \frac{\text{(unit of dislocation)}}{\text{(unit of X,Y,D)}\cdot\text{(area)}}$
- `U31, U32` : _torch.Tensor_
    - Tilt. 
    $\text{unit} = \frac{\text{(unit of dislocation)}}{\text{(unit of X,Y,D)}\cdot\text{(area)}}$


The shape of each tensor is same as that of `X,Y`.












### Examples


<!-- 単一観測点における変位と歪みを計算するには、次のようにします。
```python


ALP = 0.5
X = torch.tensor(0.0)
Y = torch.tensor(0.0)
D = 


SPOINT(ALP, X, Y, D, SD, CD, DISL1, DISL2, DISL3, compute_strain=True)

```


複数観測点における変位と歪みを計算する場合には、xとyを1次元または2次元のテンソルにするだけです。もちろん、xとyのdimとshapeは一致していなければなりません。
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


```  -->








    
## `SRECTF`
Calculate surface displacements, strains and tilts due to rectangular fault in a half-space.

### Inputs

- `ALP` : _float or torch.Tensor_
    - Medium constant, equal to $\frac{\mu}{\lambda+\mu}$.
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

If `compute_strain` is `True`, return is a list of 3 displacements and 6 spatial derivatives: \
`[U1, U2, U3, U11, U12, U21, U22, U31, U32]`

If `False`, return is a list of 3 displacements only:
`[U1, U2, U3]`

- `U1, U2, U3` : _torch.Tensor_
    - Displacement. 
    $\text{unit} = \text{(unit of dislocation)}$
- `U11, U12, U21, U22` : _torch.Tensor_
    - Strain. 
    $\text{unit} = \frac{\text{(unit of dislocation)}}{\text{(unit of X,Y, ... , AW)}}$
- `U31, U32` : _torch.Tensor_
    - Tilt. 
    $\text{unit} = \frac{\text{(unit of dislocation)}}{\text{(unit of X,Y, ... , AW)}}$


The shape of each tensor is same as that of `X,Y`.



<!-- ### Examples


単一観測点における変位と歪みを計算するには、次のようにします。


複数観測点における変位と歪みを計算する場合には、xとyを1次元または2次元のテンソルにするだけです。もちろん、xとyのdimとshapeは一致していなければなりません。

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





---

- [Back to README.md](../README.md)
- [Go to "How to use `DC3D0` and `DC3D`"](./Okada1992.md)
- [Go to "How to use `OkadaWrapper` Class"](./OkadaWrapper.md)