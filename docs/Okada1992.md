# `DC3D0`(_ALPHA, X, Y, Z, DEPTH, DIP, POT1, POT2, POT3, POT4, compute_strain=True, is_degree=True_)


Calculate displacement and strain at depth due to buried point source in a semiinfinite medium.

## Inputs

- `ALPHA` : _float or torch.Tensor_
    - Medium constant, equal to $\frac{\lambda+\mu}{\lambda+2\mu}$.
- `X, Y, Z` : _torch.Tensor_
    - Coordinate of observing point.
- `DEPTH` : _float or torch.Tensor_
    - Source depth.
- `DIP` : _torch.Tensor_
    - Dip-angle.
- `POT1, POT2, POT3, POT4` : _float or torch.Tensor_
    - Strike-, dip-, tensile- and inflate-potency.
    - $\text{potency} = \frac{\text{(moment of double-couple)}}{\mu}$ for `POT1`, `POT2`.
    - $\text{potency} = \frac{\text{(intensity of isotropic part)}}{\lambda}$ for `POT3`.
    - $\text{potency} = \frac{\text{(intensity of linear dipole)}}{\mu}$ for `POT4`.
- `compute_strain` : _bool, dafault True_
    - Option to calculate the spatial derivative of the displacement. 
    New in the PyTorch implementation.
- `is_degree` : _bool, dafault True_
    - Flag if `DIP` is in degree or not (= in radian). 
    New in the PyTorch implementation.


## Outputs

- U : _list of torch.Tensor_
    - If `compute_strain` is `True`, return is a list of 3 displacements and 9 spatial derivatives: \
    `[UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]`
    - If `False`, return is a list of 3 displacements only:
    `[UX, UY, UZ]`

    - `UX, UY, UZ` : _torch.Tensor_
        - Displacement. 
        $\text{unit} = \frac{\text{(unit of potency)}}{\text{(unit of X,Y,Z,DEPTH)}^2}$
    - `UXX, UYX, UZX` : _torch.Tensor_
        - X-derivative. 
        $\text{unit} = \frac{\text{(unit of potency)}}{\text{(unit of X,Y,Z,DEPTH)}^3}$
    - `UXY, UYY, UZY` : _torch.Tensor_
        - Y-derivative. 
        $\text{unit} = \frac{\text{(unit of potency)}}{\text{(unit of X,Y,Z,DEPTH)}^3}$
    - `UXZ, UYZ, UZZ` : _torch.Tensor_
        - Z-derivative. 
        $\text{unit} = \frac{\text{(unit of potency)}}{\text{(unit of X,Y,Z,DEPTH)}^3}$
    - The shape of each tensor is same as that of `X,Y,Z`.

- IRET : _torch.Tensor (int)_
    - Return code. 
    The shape is same as that of `X,Y,Z`, i.e., IRET is returned for each station.
    - `IRET=0` means normal, `IRET=1` means singular, `IRET=2` means positive z was given.






## Examples 

First, load the necessary libraries.

```python
import numpy as np
import torch
from OkadaTorch import DC3D0
```

To calculate the displacement and strain at a single station, do the following.
```python
ALPHA = 2.0/3.0
X = torch.tensor(0.5)
Y = torch.tensor(-0.5)
Z = torch.tensor(-0.1)

DEPTH = 2.0
DIP = torch.tensor(45.0)
POT1, POT2, POT3, POT4 = 2.0, 1.0, 0.0, 0.0

out, IRET = DC3D0(ALPHA, X, Y, Z, DEPTH, DIP, POT1, POT2, POT3, POT4, True, True)
out
```
`out` should be approximately as follows.
```
[tensor(0.0263),
 tensor(-0.0188),
 tensor(0.0865),
 tensor(0.0389),
 tensor(-0.0121),
 tensor(0.0301),
 tensor(0.0068),
 tensor(0.0235),
 tensor(0.0452),
 tensor(-0.0310),
 tensor(-0.0249),
 tensor(-0.0223)]
```
The elements of the list are `ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz`.
If `compute_strain=False`, the length of the list is 3.

And `IRET` should be as follows.
```
tensor(0, dtype=torch.int32)
```



By making `X`, `Y` and `Z` tensors of dim>0 instead of scalars, displacement and strain at multiple stations can be calculated. 
Of course, the dim and shape of `X`, `Y` and `Z` must be same.
For example, to calculate the displacement and strain on grid, do the following.
```python
ALPHA = 2.0/3.0
x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
z = np.linspace(-1, 0, 51)
X, Y, Z = np.meshgrid(x, y, z)
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)
Z = torch.from_numpy(Z)

DEPTH = 2.0
DIP = torch.tensor(45.0)
POT1, POT2, POT3, POT4 = 2.0, 1.0, 0.0, 0.0

out, IRET = DC3D0(ALPHA, X, Y, Z, DEPTH, DIP, POT1, POT2, POT3, POT4, True, True)
``` 
`out` should be approximately as follows.
```
[tensor([[[ 0.0508,  0.0498,  0.0488,  ...,  0.0151,  0.0146,  0.0142],
          [ 0.0510,  0.0499,  0.0489,  ...,  0.0147,  0.0142,  0.0137],
          [ 0.0511,  0.0501,  0.0490,  ...,  0.0143,  0.0138,  0.0133],
          ...,
          [ 0.0510,  0.0504,  0.0498,  ...,  0.0335,  0.0335,  0.0337],
          [ 0.0509,  0.0503,  0.0497,  ...,  0.0340,  0.0341,  0.0342],
          [ 0.0507,  0.0502,  0.0496,  ...,  0.0345,  0.0346,  0.0348]],
 
         [[ 0.0515,  0.0504,  0.0493,  ...,  0.0151,  0.0146,  0.0141],
          [ 0.0516,  0.0506,  0.0495,  ...,  0.0147,  0.0142,  0.0137],
          [ 0.0518,  0.0507,  0.0496,  ...,  0.0143,  0.0137,  0.0132],
          ...,
          [ 0.0520,  0.0514,  0.0508,  ...,  0.0339,  0.0340,  0.0341],
          [ 0.0519,  0.0513,  0.0507,  ...,  0.0344,  0.0345,  0.0347],
          [ 0.0517,  0.0512,  0.0506,  ...,  0.0349,  0.0351,  0.0352]],
 
         [[ 0.0521,  0.0510,  0.0499,  ...,  0.0150,  0.0145,  0.0141],
          [ 0.0523,  0.0512,  0.0501,  ...,  0.0146,  0.0141,  0.0136],
          [ 0.0525,  0.0513,  0.0502,  ...,  0.0142,  0.0137,  0.0131],
          ...,
          [ 0.0531,  0.0525,  0.0518,  ...,  0.0343,  0.0344,  0.0346],
          [ 0.0530,  0.0523,  0.0517,  ...,  0.0348,  0.0350,  0.0351],
          [ 0.0528,  0.0522,  0.0516,  ...,  0.0353,  0.0355,  0.0357]],
 
         ...,
...
          ...,
          [ 0.0246,  0.0240,  0.0234,  ..., -0.0019, -0.0019, -0.0019],
          [ 0.0242,  0.0236,  0.0230,  ..., -0.0018, -0.0018, -0.0018],
          [ 0.0237,  0.0232,  0.0226,  ..., -0.0017, -0.0017, -0.0017]]],
        dtype=torch.float64)]
```
To convert each tensor to a `Numpy` array, do the following.
```python
ux = out[0].detach().numpy()
```

In this case, all components of `IRET` is zero.
```python
IRET.sum() # -> tensor(0)
```





# `DC3D`(_ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3, compute_strain=True, is_degree=True_)

Calculate displacement and strain at depth due to buried finite fault in a semiinfinite medium.

## Inputs

- `ALPHA` : _float or torch.Tensor_
    - Medium constant, equal to $\frac{\lambda+\mu}{\lambda+2\mu}$.
- `X, Y, Z` : _torch.Tensor_
    - Coordinate of observing point.
- `DEPTH` : _float or torch.Tensor_
    - Depth of reference point.
- `DIP` : _torch.Tensor_
    - Dip-angle.
- `AL1, AL2` : _float or torch.Tensor_
    - Fault length range.
- `AW1, AW2` : _float or torch.Tensor_
    - Fault width range.
- `DISL1, DISL2, DISL3` : _float or torch.Tensor_
    - Strike-, dip-, tensile-dislocations.
- `compute_strain` : _bool, dafault True_
    - Option to calculate the spatial derivative of the displacement.
    New in the PyTorch implementation.
- `is_degree` : _bool, dafault True_
    - Flag if `DIP` is in degree or not (= in radian). 
    New in the PyTorch implementation.


## Outputs

- U : _list of torch.Tensor_
    - If `compute_strain` is `True`, return is a list of 3 displacements and 9 spatial derivatives:
    `[UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]`
    - If `False`, return is a list of 3 displacements only:
    `[UX, UY, UZ]`

    - `UX, UY, UZ` : _torch.Tensor_
        - Displacement. 
        $\text{unit} = \text{(unit of dislocation)}$
    - `UXX, UYX, UZX` : _torch.Tensor_
        - X-derivative. 
        $\text{unit} = \frac{\text{(unit of dislocation)}}{\text{(unit of X,Y,Z,DEPTH,AL,AW)}}$
    - `UXY, UYY, UZY` : _torch.Tensor_
        - Y-derivative. 
        $\text{unit} = \frac{\text{(unit of dislocation)}}{\text{(unit of X,Y,Z,DEPTH,AL,AW)}}$
    - `UXZ, UYZ, UZZ` : _torch.Tensor_
        - Z-derivative. 
        $\text{unit} = \frac{\text{(unit of dislocation)}}{\text{(unit of X,Y,Z,DEPTH,AL,AW)}}$
    - The shape of each tensor is same as that of `X,Y,Z`.

- IRET : _torch.Tensor (int)_
    - Return code. 
    The shape is same as that of `X,Y,Z`, i.e., IRET is returned for each station.
    - `IRET=0` means normal, `IRET=1` means singular, `IRET=2` means positive z was given.




## Examples 

```python
import numpy as np
import torch
from OkadaTorch import DC3D
```

To calculate the displacement and strain at a single station, do the following.
```python
ALPHA = 2.0/3.0
X = torch.tensor(0.5)
Y = torch.tensor(-0.5)
Z = torch.tensor(-0.1)

DEPTH = 2.0
DIP = torch.tensor(45.0)
AL1, AL2 = -0.2, 0.2
AW1, AW2 = -0.1, 0.1

DISL1, DISL2, DISL3 = 2.0, 1.0, 0.0

out, IRET = DC3D(ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3, True, True)
out
```
`out` should be approximately as follows.
```
[tensor(0.0021),
 tensor(-0.0015),
 tensor(0.0068),
 tensor(0.0030),
 tensor(-0.0010),
 tensor(0.0024),
 tensor(0.0005),
 tensor(0.0018),
 tensor(0.0036),
 tensor(-0.0025),
 tensor(-0.0020),
 tensor(-0.0017)]
```
And `IRET` should be as follows.
```
tensor(0, dtype=torch.int32)
```



To calculate the displacement and strain on grid, do the following.
```python
ALPHA = 2.0/3.0
x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
z = np.linspace(-1, 0, 51)
X, Y, Z = np.meshgrid(x, y, z)
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)
Z = torch.from_numpy(Z)

DEPTH = 2.0
DIP = torch.tensor(45.0)
AL1, AL2 = -0.2, 0.2
AW1, AW2 = -0.1, 0.1

DISL1, DISL2, DISL3 = 2.0, 1.0, 0.0

out, IRET = DC3D(ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3, True, True)
``` 
`out` should be approximately as follows.
```
[tensor([[[ 0.0040,  0.0039,  0.0039,  ...,  0.0012,  0.0012,  0.0011],
          [ 0.0040,  0.0040,  0.0039,  ...,  0.0012,  0.0011,  0.0011],
          [ 0.0040,  0.0040,  0.0039,  ...,  0.0011,  0.0011,  0.0011],
          ...,
          [ 0.0040,  0.0040,  0.0039,  ...,  0.0027,  0.0027,  0.0027],
          [ 0.0040,  0.0040,  0.0039,  ...,  0.0027,  0.0027,  0.0027],
          [ 0.0040,  0.0040,  0.0039,  ...,  0.0027,  0.0027,  0.0028]],
 
         [[ 0.0041,  0.0040,  0.0039,  ...,  0.0012,  0.0012,  0.0011],
          [ 0.0041,  0.0040,  0.0039,  ...,  0.0012,  0.0011,  0.0011],
          [ 0.0041,  0.0040,  0.0039,  ...,  0.0011,  0.0011,  0.0011],
          ...,
          [ 0.0041,  0.0041,  0.0040,  ...,  0.0027,  0.0027,  0.0027],
          [ 0.0041,  0.0041,  0.0040,  ...,  0.0027,  0.0027,  0.0027],
          [ 0.0041,  0.0041,  0.0040,  ...,  0.0028,  0.0028,  0.0028]],
 
         [[ 0.0041,  0.0040,  0.0040,  ...,  0.0012,  0.0012,  0.0011],
          [ 0.0041,  0.0041,  0.0040,  ...,  0.0012,  0.0011,  0.0011],
          [ 0.0042,  0.0041,  0.0040,  ...,  0.0011,  0.0011,  0.0010],
          ...,
          [ 0.0042,  0.0042,  0.0041,  ...,  0.0027,  0.0027,  0.0027],
          [ 0.0042,  0.0041,  0.0041,  ...,  0.0028,  0.0028,  0.0028],
          [ 0.0042,  0.0041,  0.0041,  ...,  0.0028,  0.0028,  0.0028]],
 
         ...,
...
          ...,
          [ 0.0020,  0.0019,  0.0019,  ..., -0.0002, -0.0002, -0.0002],
          [ 0.0019,  0.0019,  0.0018,  ..., -0.0001, -0.0001, -0.0001],
          [ 0.0019,  0.0019,  0.0018,  ..., -0.0001, -0.0001, -0.0001]]],
        dtype=torch.float64)]
```

In this case, all components of `IRET` is zero.
```python
IRET.sum() # -> tensor(0)
```




# Remark

If you are familiar with the `dc3d0wrapper` or `dc3dwrapper` functions from [`okada_wrapper`](https://github.com/cutde-org/okada_wrapper), you can use similar interfaces by defining the following functions.
```python
import torch
from OkadaTorch import DC3D0, DC3D


def dc3d0wrapper(alpha, xo, depth, dip, potency):
    out, IRET = \
    DC3D0(alpha, xo[0], xo[1], xo[2], depth, dip, 
          potency[0], potency[1], potency[2], potency[3], True, True)
    return IRET, out[:3], out[3:]

def dc3dwrapper(alpha, xo, depth, dip, strike_width, dip_width, dislocation):
    out, IRET = \
    DC3D(alpha, xo[0], xo[1], xo[2], depth, dip, 
         strike_width[0], strike_width[1], dip_width[0], dip_width[1],
         dislocation[0], dislocation[1], dislocation[2], True, True)
    return IRET, out[:3], out[3:]
```

To obtain the same results as `test_dc3d0()` and `test_dc3d()` in [`test_okada.py`](https://github.com/cutde-org/okada_wrapper/blob/master/test_okada.py) of `okada_wrapper`, do the following.

```python
from numpy import linspace, zeros, log, meshgrid
from matplotlib.pyplot import contourf, contour, \
    xlabel, ylabel, colorbar, show, savefig
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['lines.linewidth'] = 1

def get_params():
    source_depth = 3.0
    obs_depth = 3.0
    poisson_ratio = 0.25
    mu = 1.0
    dip = 90
    lmda = 2 * mu * poisson_ratio / (1 - 2 * poisson_ratio)
    alpha = (lmda + mu) / (lmda + 2 * mu)
    return source_depth, obs_depth, poisson_ratio, mu, dip, alpha


def test_dc3d0():
    source_depth, obs_depth, poisson_ratio, mu, dip, alpha = get_params()
    n = (100, 100)
    x = linspace(-1, 1, n[0])
    y = linspace(-1, 1, n[1])
    X, Y = meshgrid(x, y)
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    success, u, grad_u = dc3d0wrapper(alpha, 
                                      [X, Y, torch.tensor(-obs_depth)], 
                                      source_depth, torch.tensor(dip), 
                                      [1.0, 0.0, 0.0, 0.0])
    
    ux = u[0]

    cntrf = contourf(x, y, log(abs(ux))) # not ux.T
    contour(x, y, log(abs(ux)), colors='k', linestyles='solid')
    xlabel(r'$\mathrm{x}$')
    ylabel(r'$\mathrm{y}$')
    cbar = colorbar(cntrf)
    cbar.set_label(r'$\log(u_{\mathrm{x}})$')
    show()


def test_dc3d():
    source_depth, obs_depth, poisson_ratio, mu, dip, alpha = get_params()
    n = (100, 100)
    x = linspace(-1, 1, n[0])
    y = linspace(-1, 1, n[1])
    X, Y = meshgrid(x, y)
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    success, u, grad_u = dc3dwrapper(alpha, 
                                     [X, Y, torch.tensor(-obs_depth)],
                                     source_depth, torch.tensor(dip),
                                     [-0.6, 0.6], [-0.6, 0.6],
                                     [1.0, 0.0, 0.0])

    ux = u[0]

    levels = linspace(-0.5, 0.5, 21)
    cntrf = contourf(x, y, ux, levels=levels) # not ux.T
    contour(x, y, ux, colors='k', levels=levels, linestyles='solid')
    xlabel(r'$\mathrm{x}$')
    ylabel(r'$\mathrm{y}$')
    cbar = colorbar(cntrf)
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.set_label(r'$u_{\mathrm{x}}$')
    savefig("strike_slip.png")
    show()


if __name__ == '__main__':
    test_dc3d0()
    test_dc3d()
```



---

- [Back to README.md](../README.md)
- [Go to the document of `SPOINT` and `SRECTF`](./Okada1985.md)
- [Go to the document of `OkadaWrapper`](./OkadaWrapper.md)