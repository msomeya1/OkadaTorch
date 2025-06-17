# `SPOINT`(_ALP, X, Y, D, SD, CD, DISL1, DISL2, DISL3, compute_strain=True_)

Calculate surface displacement, strain, tilt due to buried point source in a semiinfinite medium.


## Inputs

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

## Outputs

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












## Examples


> [!NOTE]
> We have prepared a [notebook](../notebooks/1_SPOINT_SRECTF.ipynb) to run the following Python code.


First, load the necessary libraries.

```python
import numpy as np
import torch
from OkadaTorch import SPOINT
```

To calculate the displacement and strain at a single station, do the following.
```python
ALP = 0.5
X = torch.tensor(0.5)
Y = torch.tensor(-0.5)

D = 0.05
dip = torch.tensor(45.0)
SD = torch.sin(torch.deg2rad(dip))
CD = torch.cos(torch.deg2rad(dip))

DISL1, DISL2, DISL3 = 2.0, 1.0, 0.0

out = SPOINT(ALP, X, Y, D, SD, CD, DISL1, DISL2, DISL3, True)
```
`out` should be approximately as follows.
```
[tensor(0.3715),
 tensor(-0.2740),
 tensor(-0.1631),
 tensor(-0.4587),
 tensor(1.1333),
 tensor(0.3721),
 tensor(-0.8585),
 tensor(0.1561),
 tensor(-0.3647)]
```
The elements of the list are `ux, uy, uz, uxx, uxy, uyx, uyy, uzx, uzy`.
If `compute_strain=False`, the length of the list is 3.


By making `X` and `Y` tensors of dim>0 instead of scalars, displacement and strain at multiple stations can be calculated. 
Of course, the dim and shape of `X` and `Y` must be same.
For example, to calculate the displacement and strain on grid, do the following.
```python
ALP = 0.5
x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
X, Y = np.meshgrid(x, y)
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

D = 0.05
dip = torch.tensor(45.0)
SD = torch.sin(torch.deg2rad(dip))
CD = torch.cos(torch.deg2rad(dip))

DISL1, DISL2, DISL3 = 2.0, 1.0, 0.0

out = SPOINT(ALP, X, Y, D, SD, CD, DISL1, DISL2, DISL3, True)
``` 
`out` should be approximately as follows.
```
[tensor([[ 0.1296,  0.1321,  0.1346,  ...,  0.0882,  0.0870,  0.0859],
         [ 0.1324,  0.1351,  0.1378,  ...,  0.0908,  0.0896,  0.0884],
         [ 0.1353,  0.1381,  0.1410,  ...,  0.0935,  0.0922,  0.0909],
         ...,
         [-0.0777, -0.0789, -0.0801,  ..., -0.1275, -0.1248, -0.1221],
         [-0.0759, -0.0770, -0.0780,  ..., -0.1250, -0.1225, -0.1199],
         [-0.0740, -0.0750, -0.0760,  ..., -0.1225, -0.1201, -0.1177]],
        dtype=torch.float64),
 tensor([[ 0.1558,  0.1602,  0.1648,  ..., -0.0608, -0.0603, -0.0597],
         [ 0.1577,  0.1624,  0.1671,  ..., -0.0630, -0.0624, -0.0617],
         [ 0.1596,  0.1644,  0.1693,  ..., -0.0652, -0.0644, -0.0636],
         ...,
         [ 0.0510,  0.0514,  0.0518,  ..., -0.1559, -0.1514, -0.1470],
         [ 0.0494,  0.0497,  0.0500,  ..., -0.1541, -0.1497, -0.1455],
         [ 0.0478,  0.0480,  0.0482,  ..., -0.1522, -0.1480, -0.1439]],
        dtype=torch.float64),
 tensor([[ 0.0453,  0.0464,  0.0476,  ..., -0.0499, -0.0493, -0.0488],
         [ 0.0457,  0.0469,  0.0481,  ..., -0.0512, -0.0506, -0.0500],
         [ 0.0462,  0.0474,  0.0486,  ..., -0.0526, -0.0519, -0.0513],
         ...,
         [-0.0519, -0.0526, -0.0533,  ...,  0.0493,  0.0481,  0.0468],
         [-0.0507, -0.0513, -0.0519,  ...,  0.0488,  0.0476,  0.0464],
         [-0.0494, -0.0500, -0.0505,  ...,  0.0482,  0.0470,  0.0459]],
        dtype=torch.float64),
 tensor([[ 0.1259,  0.1255,  0.1248,  ..., -0.0544, -0.0562, -0.0578],
...
         ...,
         [ 0.0645,  0.0675,  0.0706,  ..., -0.0269, -0.0242, -0.0216],
         [ 0.0638,  0.0667,  0.0696,  ..., -0.0283, -0.0256, -0.0230],
         [ 0.0630,  0.0657,  0.0685,  ..., -0.0294, -0.0268, -0.0243]],
        dtype=torch.float64)]
```
To convert each tensor to a `Numpy` array, do the following.
```python
ux = out[0].detach().numpy()
```








    
# `SRECTF`(ALP, X, Y, DEP, AL, AW, SD, CD, DISL1, DISL2, DISL3, compute_strain=True)
Calculate surface displacements, strains and tilts due to rectangular fault in a half-space.

## Inputs

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

## Outputs

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


## Examples

> [!NOTE]
> We have prepared a [notebook](../notebooks/1_SPOINT_SRECTF.ipynb) to run the following Python code.

```python
import numpy as np
import torch
from OkadaTorch import SRECTF
```

To calculate the displacement and strain at a single station, do the following.
```python
ALP = 0.5
X = torch.tensor(0.5)
Y = torch.tensor(-0.5)

DEP = 0.05
AL, AW = 0.5, 0.2
dip = torch.tensor(45.0)
SD = torch.sin(torch.deg2rad(dip))
CD = torch.cos(torch.deg2rad(dip))

DISL1, DISL2, DISL3 = 2.0, 1.0, 0.0

out = SRECTF(ALP, X, Y, DEP, AL, AW, SD, CD, DISL1, DISL2, DISL3, True)
```
`out` should be approximately as follows.
```
[tensor(0.0304),
 tensor(0.0108),
 tensor(-0.0104),
 tensor(-0.0232),
 tensor(0.1006),
 tensor(-0.1467),
 tensor(-0.0046),
 tensor(-0.0674),
 tensor(-0.0522)]
```


To calculate the displacement and strain on grid, do the following.
```python
ALP = 0.5
x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
X, Y = np.meshgrid(x, y)
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

DEP = 0.05
AL, AW = 0.5, 0.2
dip = torch.tensor(45.0)
SD = torch.sin(torch.deg2rad(dip))
CD = torch.cos(torch.deg2rad(dip))

DISL1, DISL2, DISL3 = 2.0, 1.0, 0.0

out = SRECTF(ALP, X, Y, D, SD, CD, DISL1, DISL2, DISL3, True)
``` 
`out` should be approximately as follows.
```
[tensor([[ 0.0086,  0.0088,  0.0089,  ...,  0.0079,  0.0079,  0.0078],
         [ 0.0087,  0.0089,  0.0091,  ...,  0.0082,  0.0081,  0.0081],
         [ 0.0088,  0.0090,  0.0092,  ...,  0.0085,  0.0084,  0.0084],
         ...,
         [-0.0071, -0.0073, -0.0074,  ..., -0.0180, -0.0177, -0.0174],
         [-0.0070, -0.0071, -0.0073,  ..., -0.0174, -0.0172, -0.0169],
         [-0.0069, -0.0070, -0.0072,  ..., -0.0169, -0.0166, -0.0164]],
        dtype=torch.float64),
 tensor([[ 0.0103,  0.0105,  0.0108,  ..., -0.0025, -0.0027, -0.0029],
         [ 0.0103,  0.0106,  0.0109,  ..., -0.0027, -0.0029, -0.0031],
         [ 0.0104,  0.0107,  0.0110,  ..., -0.0029, -0.0031, -0.0033],
         ...,
         [ 0.0050,  0.0051,  0.0052,  ..., -0.0248, -0.0242, -0.0235],
         [ 0.0049,  0.0050,  0.0051,  ..., -0.0242, -0.0236, -0.0230],
         [ 0.0048,  0.0049,  0.0049,  ..., -0.0236, -0.0230, -0.0225]],
        dtype=torch.float64),
 tensor([[ 0.0042,  0.0043,  0.0044,  ..., -0.0053, -0.0053, -0.0054],
         [ 0.0042,  0.0043,  0.0044,  ..., -0.0055, -0.0056, -0.0056],
         [ 0.0042,  0.0044,  0.0045,  ..., -0.0058, -0.0058, -0.0058],
         ...,
         [-0.0054, -0.0055, -0.0056,  ...,  0.0102,  0.0099,  0.0097],
         [-0.0053, -0.0054, -0.0055,  ...,  0.0099,  0.0097,  0.0094],
         [-0.0052, -0.0053, -0.0054,  ...,  0.0096,  0.0094,  0.0092]],
        dtype=torch.float64),
 tensor([[ 0.0088,  0.0089,  0.0090,  ..., -0.0025, -0.0026, -0.0028],
...
         ...,
         [ 0.0045,  0.0048,  0.0051,  ..., -0.0145, -0.0135, -0.0126],
         [ 0.0046,  0.0049,  0.0052,  ..., -0.0141, -0.0133, -0.0124],
         [ 0.0046,  0.0049,  0.0052,  ..., -0.0138, -0.0130, -0.0122]],
        dtype=torch.float64)]
```





---

- [Back to README.md](../README.md)
- [Go to the document of `DC3D0` and `DC3D`](./Okada1992.md)
- [Go to the document of `OkadaWrapper`](./OkadaWrapper.md)