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
- `compute_strain` : _bool, default True_
    - Option to calculate the spatial derivative of the displacement. 
    New in the PyTorch implementation.
- `is_degree` : _bool, default True_
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

> [!NOTE]
> We have prepared a [notebook](../2_DC3D0_DC3D.ipynb) to run the following Python code.


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

POT1, POT2, POT3, POT4 = 4.0, 3.0, 0.0, 0.0

out, IRET = DC3D0(ALPHA, X, Y, Z, DEPTH, DIP, POT1, POT2, POT3, POT4, compute_strain=True, is_degree=True)
```
`out` should be approximately as follows.
```
[tensor(0.0624),
 tensor(-0.0456),
 tensor(0.2197),
 tensor(0.0927),
 tensor(-0.0199),
 tensor(0.0337),
 tensor(0.0207),
 tensor(0.0565),
 tensor(0.1269),
 tensor(-0.0452),
 tensor(-0.0738),
 tensor(-0.0531)]
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

POT1, POT2, POT3, POT4 = 4.0, 3.0, 0.0, 0.0

out, IRET = DC3D0(ALPHA, X, Y, Z, DEPTH, DIP, POT1, POT2, POT3, POT4, compute_strain=True, is_degree=True)
``` 
`out` should be approximately as follows.
```
[tensor([[[ 0.1017,  0.0994,  0.0971,  ...,  0.0205,  0.0193,  0.0181],
          [ 0.1021,  0.0997,  0.0974,  ...,  0.0198,  0.0185,  0.0172],
          [ 0.1023,  0.0999,  0.0976,  ...,  0.0190,  0.0177,  0.0163],
          ...,
          [ 0.1019,  0.1010,  0.1000,  ...,  0.0765,  0.0770,  0.0775],
          [ 0.1016,  0.1008,  0.0999,  ...,  0.0776,  0.0781,  0.0787],
          [ 0.1014,  0.1005,  0.0996,  ...,  0.0786,  0.0792,  0.0798]],
 
         [[ 0.1028,  0.1004,  0.0981,  ...,  0.0202,  0.0190,  0.0177],
          [ 0.1032,  0.1007,  0.0984,  ...,  0.0195,  0.0181,  0.0168],
          [ 0.1035,  0.1010,  0.0986,  ...,  0.0187,  0.0173,  0.0160],
          ...,
          [ 0.1042,  0.1032,  0.1022,  ...,  0.0776,  0.0781,  0.0787],
          [ 0.1039,  0.1030,  0.1020,  ...,  0.0787,  0.0792,  0.0799],
          [ 0.1036,  0.1027,  0.1018,  ...,  0.0797,  0.0803,  0.0810]],
 
         [[ 0.1039,  0.1014,  0.0990,  ...,  0.0199,  0.0186,  0.0173],
          [ 0.1043,  0.1018,  0.0993,  ...,  0.0191,  0.0178,  0.0164],
          [ 0.1046,  0.1021,  0.0996,  ...,  0.0184,  0.0170,  0.0156],
          ...,
          [ 0.1066,  0.1055,  0.1045,  ...,  0.0787,  0.0792,  0.0798],
          [ 0.1063,  0.1053,  0.1042,  ...,  0.0798,  0.0804,  0.0810],
          [ 0.1059,  0.1049,  0.1039,  ...,  0.0808,  0.0815,  0.0822]],
 
         ...,
...
          ...,
          [ 0.0626,  0.0611,  0.0596,  ..., -0.0036, -0.0036, -0.0035],
          [ 0.0613,  0.0599,  0.0584,  ..., -0.0033, -0.0033, -0.0032],
          [ 0.0600,  0.0587,  0.0572,  ..., -0.0030, -0.0030, -0.0029]]],
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
- `compute_strain` : _bool, default True_
    - Option to calculate the spatial derivative of the displacement.
    New in the PyTorch implementation.
- `is_degree` : _bool, default True_
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

> [!NOTE]
> We have prepared a [notebook](../2_DC3D0_DC3D.ipynb) to run the following Python code.


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

DISL1, DISL2, DISL3 = 4.0, 3.0, 0.0

out, IRET = DC3D(ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3, compute_strain=True, is_degree=True)
```
`out` should be approximately as follows.
```
[tensor(0.0050),
 tensor(-0.0036),
 tensor(0.0174),
 tensor(0.0072),
 tensor(-0.0016),
 tensor(0.0028),
 tensor(0.0016),
 tensor(0.0044),
 tensor(0.0101),
 tensor(-0.0036),
 tensor(-0.0059),
 tensor(-0.0041)]
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

DISL1, DISL2, DISL3 = 4.0, 3.0, 0.0

out, IRET = DC3D(ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3, compute_strain=True, is_degree=True)
``` 
`out` should be approximately as follows.
```
[tensor([[[ 0.0081,  0.0079,  0.0077,  ...,  0.0016,  0.0015,  0.0014],
          [ 0.0081,  0.0079,  0.0077,  ...,  0.0016,  0.0015,  0.0014],
          [ 0.0081,  0.0079,  0.0077,  ...,  0.0015,  0.0014,  0.0013],
          ...,
          [ 0.0081,  0.0080,  0.0079,  ...,  0.0061,  0.0061,  0.0061],
          [ 0.0081,  0.0080,  0.0079,  ...,  0.0062,  0.0062,  0.0062],
          [ 0.0080,  0.0080,  0.0079,  ...,  0.0062,  0.0063,  0.0063]],
 
         [[ 0.0081,  0.0080,  0.0078,  ...,  0.0016,  0.0015,  0.0014],
          [ 0.0082,  0.0080,  0.0078,  ...,  0.0016,  0.0015,  0.0013],
          [ 0.0082,  0.0080,  0.0078,  ...,  0.0015,  0.0014,  0.0013],
          ...,
          [ 0.0083,  0.0082,  0.0081,  ...,  0.0062,  0.0062,  0.0062],
          [ 0.0082,  0.0082,  0.0081,  ...,  0.0062,  0.0063,  0.0063],
          [ 0.0082,  0.0081,  0.0081,  ...,  0.0063,  0.0064,  0.0064]],
 
         [[ 0.0082,  0.0080,  0.0078,  ...,  0.0016,  0.0015,  0.0014],
          [ 0.0083,  0.0081,  0.0079,  ...,  0.0015,  0.0014,  0.0013],
          [ 0.0083,  0.0081,  0.0079,  ...,  0.0015,  0.0014,  0.0013],
          ...,
          [ 0.0084,  0.0084,  0.0083,  ...,  0.0062,  0.0063,  0.0063],
          [ 0.0084,  0.0083,  0.0083,  ...,  0.0063,  0.0064,  0.0064],
          [ 0.0084,  0.0083,  0.0082,  ...,  0.0064,  0.0065,  0.0065]],
 
         ...,
...
          ...,
          [ 0.0050,  0.0049,  0.0048,  ..., -0.0003, -0.0003, -0.0003],
          [ 0.0049,  0.0048,  0.0047,  ..., -0.0003, -0.0003, -0.0003],
          [ 0.0048,  0.0047,  0.0046,  ..., -0.0002, -0.0002, -0.0002]]],
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
          potency[0], potency[1], potency[2], potency[3], compute_strain=True, is_degree=True)
    return IRET, out[:3], out[3:]

def dc3dwrapper(alpha, xo, depth, dip, strike_width, dip_width, dislocation):
    out, IRET = \
    DC3D(alpha, xo[0], xo[1], xo[2], depth, dip, 
         strike_width[0], strike_width[1], dip_width[0], dip_width[1],
         dislocation[0], dislocation[1], dislocation[2], compute_strain=True, is_degree=True)
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
    show()


if __name__ == '__main__':
    test_dc3d0()
    test_dc3d()
```



---

- [Back to README.md](../README.md)
- [Go to the document of `SPOINT` and `SRECTF`](./Okada1985.md)
- [Go to the document of `OkadaWrapper`](./OkadaWrapper.md)