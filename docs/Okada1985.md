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
> We have prepared a [notebook](../1_SPOINT_SRECTF.ipynb) to run the following Python code.


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

D = 1.0
dip = torch.tensor(45.0)
SD = torch.sin(torch.deg2rad(dip))
CD = torch.cos(torch.deg2rad(dip))

DISL1, DISL2, DISL3 = 4.0, 3.0, 0.0

out = SPOINT(ALP, X, Y, D, SD, CD, DISL1, DISL2, DISL3, compute_strain=True)
```
`out` should be approximately as follows.
```
[tensor(0.3194),
 tensor(-0.2997),
 tensor(0.5736),
 tensor(0.4528),
 tensor(0.4729),
 tensor(0.0979),
 tensor(0.0660),
 tensor(-0.2761),
 tensor(0.9807)]
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

D = 1.0
dip = torch.tensor(45.0)
SD = torch.sin(torch.deg2rad(dip))
CD = torch.cos(torch.deg2rad(dip))

DISL1, DISL2, DISL3 = 4.0, 3.0, 0.0

out = SPOINT(ALP, X, Y, D, SD, CD, DISL1, DISL2, DISL3, compute_strain=True)
``` 
`out` should be approximately as follows.
```
[tensor([[ 0.1568,  0.1559,  0.1549,  ...,  0.1966,  0.1975,  0.1982],
         [ 0.1586,  0.1577,  0.1567,  ...,  0.2028,  0.2036,  0.2042],
         [ 0.1603,  0.1594,  0.1584,  ...,  0.2091,  0.2098,  0.2103],
         ...,
         [-0.0252, -0.0258, -0.0264,  ...,  0.0242,  0.0245,  0.0248],
         [-0.0251, -0.0257, -0.0263,  ...,  0.0199,  0.0202,  0.0206],
         [-0.0250, -0.0255, -0.0260,  ...,  0.0157,  0.0161,  0.0165]],
        dtype=torch.float64),
 tensor([[ 0.1753,  0.1771,  0.1789,  ..., -0.1843, -0.1821, -0.1798],
         [ 0.1745,  0.1763,  0.1781,  ..., -0.1873, -0.1849, -0.1825],
         [ 0.1735,  0.1754,  0.1773,  ..., -0.1902, -0.1877, -0.1851],
         ...,
         [ 0.0073,  0.0075,  0.0076,  ...,  0.0054,  0.0048,  0.0042],
         [ 0.0069,  0.0070,  0.0071,  ...,  0.0021,  0.0016,  0.0011],
         [ 0.0065,  0.0066,  0.0066,  ..., -0.0012, -0.0016, -0.0020]],
        dtype=torch.float64),
 tensor([[-0.1359, -0.1376, -0.1392,  ...,  0.1642,  0.1615,  0.1588],
         [-0.1380, -0.1398, -0.1414,  ...,  0.1704,  0.1676,  0.1647],
         [-0.1401, -0.1419, -0.1436,  ...,  0.1768,  0.1737,  0.1706],
         ...,
         [-0.0145, -0.0140, -0.0135,  ...,  0.0466,  0.0458,  0.0450],
         [-0.0145, -0.0140, -0.0135,  ...,  0.0425,  0.0418,  0.0411],
         [-0.0144, -0.0140, -0.0135,  ...,  0.0386,  0.0380,  0.0374]],
        dtype=torch.float64),
 tensor([[-0.0408, -0.0482, -0.0557,  ...,  0.0487,  0.0399,  0.0314],
...
          -2.0510e-01, -1.9926e-01],
         [ 2.2898e-03,  2.5732e-04, -1.9310e-03,  ..., -2.0070e-01,
          -1.9520e-01, -1.8977e-01],
         [ 1.2462e-03, -7.1597e-04, -2.8236e-03,  ..., -1.9064e-01,
          -1.8555e-01, -1.8052e-01]], dtype=torch.float64)]
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
> We have prepared a [notebook](../1_SPOINT_SRECTF.ipynb) to run the following Python code.

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

DEP = 1.0
AL, AW = 0.5, 0.2
dip = torch.tensor(45.0)
SD = torch.sin(torch.deg2rad(dip))
CD = torch.cos(torch.deg2rad(dip))

DISL1, DISL2, DISL3 = 4.0, 3.0, 0.0

out = SRECTF(ALP, X, Y, DEP, AL, AW, SD, CD, DISL1, DISL2, DISL3, compute_strain=True)
```
`out` should be approximately as follows.
```
[tensor(0.0171),
 tensor(-0.0284),
 tensor(0.0467),
 tensor(0.0588),
 tensor(0.0325),
 tensor(-0.0367),
 tensor(-0.0299),
 tensor(0.0485),
 tensor(0.1255)]
```


To calculate the displacement and strain on grid, do the following.
```python
ALP = 0.5
x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
X, Y = np.meshgrid(x, y)
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

DEP = 1.0
AL, AW = 0.5, 0.2
dip = torch.tensor(45.0)
SD = torch.sin(torch.deg2rad(dip))
CD = torch.cos(torch.deg2rad(dip))

DISL1, DISL2, DISL3 = 4.0, 3.0, 0.0

out = SRECTF(ALP, X, Y, DEP, AL, AW, SD, CD, DISL1, DISL2, DISL3, compute_strain=True)
``` 
`out` should be approximately as follows.
```
[tensor([[ 0.0163,  0.0163,  0.0164,  ...,  0.0152,  0.0155,  0.0158],
         [ 0.0165,  0.0165,  0.0166,  ...,  0.0158,  0.0161,  0.0164],
         [ 0.0166,  0.0167,  0.0168,  ...,  0.0164,  0.0167,  0.0170],
         ...,
         [-0.0018, -0.0019, -0.0020,  ...,  0.0022,  0.0023,  0.0023],
         [-0.0019, -0.0019, -0.0020,  ...,  0.0016,  0.0017,  0.0018],
         [-0.0019, -0.0020, -0.0020,  ...,  0.0011,  0.0012,  0.0013]],
        dtype=torch.float64),
 tensor([[ 0.0163,  0.0166,  0.0169,  ..., -0.0182, -0.0183, -0.0183],
         [ 0.0163,  0.0166,  0.0168,  ..., -0.0188, -0.0188, -0.0188],
         [ 0.0162,  0.0165,  0.0168,  ..., -0.0194, -0.0194, -0.0194],
         ...,
         [ 0.0008,  0.0008,  0.0008,  ...,  0.0014,  0.0013,  0.0013],
         [ 0.0007,  0.0007,  0.0007,  ...,  0.0009,  0.0008,  0.0008],
         [ 0.0007,  0.0007,  0.0007,  ...,  0.0004,  0.0003,  0.0003]],
        dtype=torch.float64),
 tensor([[-0.0105, -0.0107, -0.0109,  ...,  0.0146,  0.0146,  0.0146],
         [-0.0107, -0.0109, -0.0111,  ...,  0.0154,  0.0154,  0.0153],
         [-0.0109, -0.0111, -0.0113,  ...,  0.0162,  0.0161,  0.0161],
         ...,
         [-0.0022, -0.0022, -0.0022,  ...,  0.0063,  0.0062,  0.0061],
         [-0.0022, -0.0021, -0.0021,  ...,  0.0056,  0.0056,  0.0055],
         [-0.0021, -0.0021, -0.0021,  ...,  0.0050,  0.0050,  0.0049]],
        dtype=torch.float64),
 tensor([[ 0.0035,  0.0030,  0.0026,  ...,  0.0158,  0.0150,  0.0143],
...
         ...,
         [ 0.0024,  0.0023,  0.0023,  ..., -0.0343, -0.0334, -0.0326],
         [ 0.0022,  0.0022,  0.0021,  ..., -0.0322, -0.0314, -0.0307],
         [ 0.0021,  0.0020,  0.0019,  ..., -0.0302, -0.0295, -0.0288]],
        dtype=torch.float64)]
```





---

- [Back to README.md](../README.md)
- [Go to the document of `DC3D0` and `DC3D`](./Okada1992.md)
- [Go to the document of `OkadaWrapper`](./OkadaWrapper.md)