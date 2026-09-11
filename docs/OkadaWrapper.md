# class `OkadaWrapper`

`OkadaWrapper` is a convenient wrapper class that can handle `SPOINT`, `SRECTF`, `DC3D0` and `DC3D` with common interface.
It also provides functions to calculate gradient and hessian.



✅ Quick Summary

| Method   | Input                    | Output                              |
| -------- | ------------------------ | ----------------------------------- |
| compute  | coords + params          | \[ux, uy, uz, ...] or \[ux, uy, uz] |
| gradient | coords + params + arg    | ∂output / ∂arg                      |
| hessian  | coords + params + arg1/2 | ∂²output / ∂arg1∂arg2               |


```python
coords = {
    "x": X,
    "y": Y,
    "z": Z
}
params = {
    "x_fault": x_fault,
    "y_fault": y_fault,
    "depth": depth,
    "length": length,
    "width": width,
    "strike": strike,
    "dip": dip,
    "rake": rake,
    "slip": slip
}

okada = OkadaWrapper()

out = okada.compute(coords, params)
grad = okada.gradient(coords, params, arg="depth")
hess = okada.hessian(coords, params, arg1="depth", arg2="dip")
```



## Introduction


The `OkadaWrapper` class has three methods (`compute`, `gradient`, and `hessian`). 
Here, their common arguments, `coords`, `params`, `compute_strain`, `is_degree`, `fault_origin` and `nu`, are explained first.


### `coords`

`coords` is a Python dictionary that stores the coordinates of stations.
Allowed keys are `"x"`, `"y"`, `"z"` and the corresponding values are `torch.Tensor`s representing the coordinates of the stations. 
In this coordinate system, x is east, y is north, and z is upward.
`z` must be non-positive. A positive `z` is **not** an error: the station is flagged with `IRET = 2` and returned as zero, exactly as the original FORTRAN does. See [Return codes](#return-codes).



### `params`

`params` is also a Python dictionary that stores the values of source parameters.
Allowed keys are `"x_fault"`, `"y_fault"`, `"depth"`, `"length"`, `"width"`, `"strike"`, `"dip"`, `"rake"`, `"slip"`, `"opening"`, `"inflation"`.
The explanation of each key is as follows.

- `x_fault, y_fault, depth`: x, y coordinates and depth of the source (depth is positive).
In the case of a point source, these values of course represent the location of that point.
In the case of a rectangular fault, the flag `fault_origin` specifies which point these values represent.
If `fault_origin` is `"topleft"`, then `x_fault`, `y_fault` and `depth` represent the coordinates of the top left corner of the rectangle.
If `fault_origin` is `"center"`, then `x_fault`, `y_fault` and `depth` represent the coordinates of the rectangle's center.
- `length, width`: These parameters are specific to a rectangular fault. `length` corresponds to the strike direction and `width` to the dip direction.
- `strike, dip, rake`: If `is_degree` is `True`, these variables are measured in degrees. If `False`, in radians. 
Note that `strike` is measured clockwise from the north and the fault is assumed to dip to the right hand side as viewed from the strike direction. 
- `slip`: This name can be misleading. 
For a rectangular fault, this indicates the amount of slip, as the name implies.
For a point source, this indicates the potency (which is equal to seismic moment divided by the elastic constant, also equal to slip amount multiplied by the infinitesimal area of the fault). 
For ease of implementation, they are treated under the same name `slip`.
- `opening`: Tensile opening, perpendicular to the fault plane, in the same units as `slip`. 
This is the third dislocation component of Okada (1985, 1992), used for a dike or a sill. 
Optional; it defaults to zero, which is the pure shear case.
- `inflation`: Isotropic volume change of a point source (an inflation source). 
Only `DC3D0` has this component, so it requires `z` and must not be combined with `length` / `width`. 
Optional; it defaults to zero.

> [!NOTE]
> `opening` and `inflation` are the only optional keys; the other nine are
> required, except that `length` and `width` must be given together or not at all
> (both = rectangular fault, neither = point source).


### `compute_strain`
The original FORTRAN subroutine computes both displacements and strains, but in some cases displacement alone may be sufficient. 
If `compute_strain` is `True` (default), both displacements and strains are computed.
If `False`, only the displacement is computed.
In this case, intermediate variables that are only used to compute strain are not assigned, thus reducing computational cost.


### `is_degree`
If `True` (default),  `"strike"`, `"dip"` and `"rake"` are in degrees. If `False`, in radians. 


### `fault_origin`
In the case of a rectangular fault, this flag specifies which point the fault location parameter refers to. It is ignored for a point source, but is still validated there, so that a typo cannot pass unnoticed.
- If `fault_origin` is `"topleft"`, then `"x_fault"`, `"y_fault"` and `"depth"` in `params` represent the coordinates of the top left corner of the rectangle.
- If `fault_origin` is `"center"`, then `"x_fault"`, `"y_fault"` and `"depth"` in `params` represent the coordinates of the rectangle's center.

Other strings cannot be specified. 


### `nu`

Poisson's ratio of the assumed medium. Default value is 0.25, which means Poisson medium.
Must satisfy `-1 < nu < 0.5`; outside that range the elastic
energy is not positive definite and the formulae degenerate (at `nu = 0.5` the
Okada 1985 medium constant `1 - 2*nu` vanishes and every term collapses to zero).









## `OkadaWrapper.compute`(_coords:dict, params:dict, compute_strain:bool=True, is_degree:bool=True, fault_origin:str="topleft", nu:float=0.25, return_iret:bool=False_)

Perform forward computations; given the source parameters, the displacements and/or their spatial derivatives at the stations are calculated.

Multiple station coordinates can be specified, but only one set of source parameters can be specified. 
For multiple sources, map this method over them with `torch.func.vmap` and sum; see [Multiple sources](#multiple-sources).


### Inputs

- `coords` : _dict of torch.Tensor_
    - `"x"` and `"y"` are required keys, and `"z"` is optional. **Unrecognized keys are rejected** (a `"Z"` typo used to switch silently to the surface formulation).
    Each value must be torch.Tensor of the same shape (`dim` is arbitrary).

- `params` : _dict of torch.Tensor_
    - `"x_fault"`, `"y_fault"`, `"depth"`, `"strike"`, `"dip"`, `"rake"` and `"slip"` are required keys.
    `"length"` and `"width"` are optional and must be given **together** (both = rectangular fault, neither = point source; supplying only one raises).
    `"opening"` and `"inflation"` are optional and default to zero; see [`params`](#params).
    **Unrecognized keys are rejected.**
    Each value must be a scalar (a 0-dim `torch.Tensor`, or a plain Python number, which is promoted for you).

- `compute_strain` : _bool, default True_
    - Option to calculate the spatial derivative of the displacement.

- `is_degree` : _bool, default True_
    - Flag if `"strike"`, `"dip"` and `"rake"` are in degree or not (= in radian). 

- `fault_origin` : _str, default "topleft"_
    - Flag if `"x_fault"`, `"y_fault"` and `"depth"` represent the location of top left corner of the rectangle or the center.

- `nu` : _float, default 0.25_
    - Poisson's ratio.





### Outputs

> [!NOTE]
> In the following, for the sake of explanation, the outputs are collectively denoted as `u`.


If `compute_strain` is `True`, `u` is a list of 3 displacements and 9 spatial derivatives: \
`[ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz]` \
i.e., 
$$\left[u_x, u_y, u_z, \frac{\partial u_x}{\partial x}, \ldots , \frac{\partial u_z}{\partial z}\right].$$
If `False`, `u` is a list of 3 displacements only:
`[ux, uy, uz]`

- `ux, uy, uz` : _torch.Tensor_
    - Displacement.
- `uxx, uyx, uzx` : _torch.Tensor_
    - x-derivative.
- `uxy, uyy, uzy` : _torch.Tensor_
    - y-derivative.
- `uxz, uyz, uzz` : _torch.Tensor_
    - z-derivative.

The shape of each tensor is same as that of `x,y(,z)`.


> [!IMPORTANT]
> In the `compute` method (and of course, `gradient` and `hessian` method), the function to be called is determined by the keys of `coords` and `params`. That is,
> - if `x, y ∈ coords` but `z ∉ coords`, and `x_fault, y_fault, depth, strike, dip, rake, slip ∈ params` but `length, width ∉ params`, then `SPOINT` is called.
> - if `x, y ∈ coords` but `z ∉ coords`, and `x_fault, y_fault, depth, length, width, strike, dip, rake, slip ∈ params`, then `SRECTF` is called.
> - if `x, y, z ∈ coords`, and `x_fault, y_fault, depth, strike, dip, rake, slip ∈ params` but `length, width ∉ params`, then `DC3D0` is called.
> - if `x, y, z ∈ coords`, and `x_fault, y_fault, depth, length, width, strike, dip, rake, slip ∈ params`, then `DC3D` is called.
>
> If the required keys are missing, or if unrecognized keys are present, a `ValueError` is raised. See [Errors](#errors).





> [!NOTE]
> By default there is no `IRET`, as in the original `SPOINT`/`SRECTF`.
> Pass `return_iret=True` to get it; see [Return codes](#return-codes).





### Examples

We have prepared a [notebook](../3_OkadaWrapper_compute.ipynb) to test the `compute` method.


> [!NOTE]
> Source parameters used in the notebooks were taken from the model 10 of Table S1 in Baba et al. 2021.
> - Baba, T., Chikasada, N., Imai, K., Tanioka, Y., & Kodaira, S., 2021. 
Frequency dispersion amplifies tsunamis caused by outer-rise normal faults, Scientific Reports, 11(1), 20064, 
doi: https://doi.org/10.1038/s41598-021-99536-x.





### Multiple sources


`compute` takes one source per call: every entry of `params` must be a scalar. 
However, realistic sources are usually described as a superposition of subfaults, 
so it is convenient to be able to handle sources in batches.

`torch.func.vmap` maps `compute` over the batch, and the sum over the batch dimension provides the superimposed solution. This stays differentiable, so a slip distribution can be inverted for in the same way a single fault can.


Here we show only minimal code examples; see the [notebook](../3_OkadaWrapper_compute.ipynb) example for the full code.


Collect the source parameters into a `(n_faults, 9)` tensor, one row per subfault
in the order `keys` gives, and map over its rows.

```python
import torch
from torch.func import vmap
from OkadaTorch import OkadaWrapper

okada = OkadaWrapper()
keys = ["x_fault", "y_fault", "depth", "length", "width",
        "strike", "dip", "rake", "slip"]

def one_fault(values):
    """Vertical displacement from a single subfault."""
    return okada.compute(coords, dict(zip(keys, values)), compute_strain=False,
                         is_degree=True, fault_origin="topleft")[2]

uz = vmap(one_fault)(faults).sum(0)
```

Here `coords` is the usual dictionary of station coordinates and `faults` is the
`(n_faults, 9)` tensor, so `vmap(one_fault)(faults)` has shape
`(n_faults, *coords["x"].shape)`. The `[2]` picks `uz` out of the returned list;
drop it to keep all three components.

In the notebook, vertical displacement at the surface generated by the 2011 Tohoku-oki earthquake (Fujii et al., 2011) are shown. Two points from that example are worth knowing in advance:


> [!NOTE]
> `vmap` is a convenience, not a requirement: a Python loop over the rows of
> `faults` gives the same answer to within rounding (5e-15 in that example).
> What `vmap` buys is one batched call instead of 40 sequential ones.
> It also maps over every source parameter, not only `slip`, so the geometry of
> each subfault can be inverted for as well.

**Reference**

- Fujii, Y., Satake, K., Sakai, S., Shinohara, M., & Kanazawa, T. (2011). Tsunami
  source of the 2011 off the Pacific coast of Tohoku earthquake. Earth, Planets
  and Space, 63(7), 815-820. https://doi.org/10.5047/eps.2011.06.010




## Return codes

Some station geometries have no solution: the station coincides with the point
source, or lies exactly on an edge of the rectangle, or sits above the free
surface. The original FORTRAN detects these and returns zeros with a non-zero
`IRET`; `OkadaWrapper` does the same, and can hand you the flag:

```python
out, iret = okada.compute(coords, params, compute_strain=False, return_iret=True)
```

| `IRET` | meaning |
| ------ | ------- |
| 0 | normal |
| 1 | singular: the station coincides with the source, or lies on a fault edge |
| 2 | a positive `z` was given |

For flagged stations, outputs are **exactly zero**, so without the flag they are
indistinguishable from a station whose displacement genuinely vanishes.



## Errors

Invalid input raises `ValueError` (not `AssertionError`). The checks are:

- required keys present in `coords` and `params`
- no unrecognized keys in either
- all coordinates the same shape
- `"length"` and `"width"` given together or not at all
- `"inflation"` only for a point source with `"z"`
- `fault_origin` is `"topleft"` or `"center"` (checked even for a point source,
  where it is otherwise ignored, so that a typo cannot pass unnoticed)
- `-1 < nu < 0.5`
- one device across all inputs

Dtypes need not match: they are promoted to the widest one supplied, as in any
torch expression. `float32` throughout is accepted but warns.









## `OkadaWrapper.gradient`(_coords:dict, params:dict, arg:str, compute_strain:bool=True, is_degree:bool=True, fault_origin:str="topleft", nu:float=0.25_)

Calculate gradient with respect to specified `arg` (one of coordinates or parameters) at the stations, given the source parameters.
PyTorch's function `jacfwd` is used internally.

> [!NOTE]
> Only a single `arg` can be specified.
> If you want to get gradient with respect to multiple args, you need to call this method multiple times.





If `"x", "y" (, "z")` is specified as `arg` (i.e., what is allowed as a key in the `coords`), the spatial derivative of `u` is calculated. 
If the component of `u` is displacement, the strain will be output. 
Since this is provided in the original Okada's formula, it is redundant to compute the strains with AD (simply using `compute` method is faster).
Note that it has been verified that the error is sufficiently small when the strain is calculated by the two methods. 
If the component of `u` is strain, it means that it is the second-order spatial derivative of the displacement, which cannot be computed with the original Okada's formula, so there is an advantage to computing it with AD.


If `"x_fault", "y_fault", "depth", "length", "width", "strike", "dip", "rake", "slip", "opening", "inflation"` is specified as `arg` (i.e., what is allowed as a key in the `params`), the derivative of `u` with respect to parameters is calculated. 
These are not provided in the original Okada's formula. 
You can implement the derivatives with respect to the parameters by calculating them manually, but this is very time-consuming and may produce errors, so using AD is a better choice.







### Inputs
- `coords` : _dict of torch.Tensor_
    - same as that of `compute` method. 

- `params` : _dict of torch.Tensor_
    - same as that of `compute` method. 

- `arg` : _str_
    - Name of the variable to be differentiated. 
    This should be a key of `coords` or `params`; 
    if there is no `"z"` in `coords`, you cannot specify `"z"` as `arg`.
    Similarly, an optional key of `params` -- `"length"`, `"width"`, `"opening"`, `"inflation"` -- can only be `arg` if you actually supplied it. 
    Pass `"opening": 0.0` to differentiate at zero opening. 

- `compute_strain` : _bool, default True_
    - same as that of `compute` method. 

- `is_degree` : _bool, default True_
    - same as that of `compute` method. 

- `fault_origin` : _str, default "topleft"_
    - Flag if `"x_fault"`, `"y_fault"` and `"depth"` represent the location of top left corner of the rectangle or the center.

- `nu` : _float, default 0.25_
    - same as that of `compute` method. 







### Outputs



If `compute_strain` is `True`, return is a list of 3 displacements and 9 spatial derivatives differentiated by `arg`: \
`[∂(ux)/∂(arg), ∂(uy)/∂(arg), ∂(uz)/∂(arg), ∂(uxx)/∂(arg), ..., ∂(uzz)/∂(arg)]` \
i.e., 
$$\left[\frac{\partial u_x}{\partial\text{(arg)}}, \frac{\partial u_y}{\partial\text{(arg)}}, \frac{\partial u_z}{\partial\text{(arg)}}, \frac{\partial}{\partial\text{(arg)}}\left(\frac{\partial u_x}{\partial x}\right), \ldots, \frac{\partial}{\partial\text{(arg)}}\left(\frac{\partial u_z}{\partial z}\right)\right].$$
If `False`, return is a list of 3 displacements differentiated by `arg`: \
`[∂(ux)/∂(arg), ∂(uy)/∂(arg), ∂(uz)/∂(arg)]` \
The shape of each tensor is same as that of `x,y(,z)`.









> [!TIP]
> `OkadaWrapper` can be used to find fault parameters that minimize a certain loss function (written in PyTorch function).
> In this case, the gradient value could be obtained explicitly by the `gradient` method and passed to the optimizer, but this would be redundant.
> Instead, it is easier to define a loss function, specify the parameters to be optimized, and then use `loss.backward()`.
> See the corresponding [notebook](../6_OkadaWrapper_optimization.ipynb) for the example.




### Examples


We have prepared a [notebook](../4_OkadaWrapper_gradient.ipynb) to test the `gradient` method.




## `OkadaWrapper.hessian`(_coords:dict, params:dict, arg1:str, arg2:str, compute_strain:bool=True, is_degree:bool=True, fault_origin:str="topleft", nu:float=0.25_)

Calculate hessian (2nd-order derivatives) with respect to specified `arg1` and `arg2` at the station, given the source parameters.
PyTorch's function `jacfwd` is used internally.

Theoretically, it is possible to differentiate `u` once by a spatial variable and once by a parameter.However, this is not implemented.
**Both `arg1` and `arg2` must be variables of the same kind; both must be `coords` or both must be `params`.**

If `"x", "y" (, "z")` is specified as `arg1` and `arg2` (i.e., what is allowed as a key in the `coords`), the second-order spatial derivative of `u` is calculated. 

If `"x_fault", "y_fault", "depth", "length", "width", "strike", "dip", "rake", "slip", "opening", "inflation"` is specified as `arg1` and `arg2` (i.e., what is allowed as a key in the `params`), the second-order derivative of `u` with respect to parameters is calculated. 

> [!NOTE]
> `u` is linear in the three dislocation components, so any second derivative
> taken only among `"slip"`, `"opening"` and `"inflation"` is exactly zero.
>  Mixed derivatives such as `arg1="slip", arg2="rake"` are not zero, 
> because `rake` divides `slip` between the strike and dip components.





### Inputs  
- `coords` : _dict of torch.Tensor_
    - same as that of `compute` method. 

- `params` : _dict of torch.Tensor_
    - same as that of `compute` method. 

- `arg1, arg2` : _str_
    - Name of the variable to be differentiated. 
    This should be a key of `coords` or `params`;
    if there is no `"z"` in `coords`, you cannot specify `"z"` as `arg1` or `arg2`.
    Similarly, an optional key of `params` -- `"length"`, `"width"`, `"opening"`, `"inflation"` -- can only be `arg1` or `arg2` if you actually supplied it. 
    

- `compute_strain` : _bool, default True_
    - same as that of `compute` method. 

- `is_degree` : _bool, default True_
    - same as that of `compute` method. 

- `fault_origin` : _str, default "topleft"_
    - Flag if `"x_fault"`, `"y_fault"` and `"depth"` represent the location of top left corner of the rectangle or the center.
    
- `nu` : _float, default 0.25_
    - same as that of `compute` method. 

   




### Outputs



If `compute_strain` is `True`, return is a list of 3 displacements and 9 spatial derivatives differentiated by `arg1` and `arg2`: \
`[∂^2(ux)/∂(arg1)∂(arg2), ∂^2(uy)/∂(arg1)∂(arg2), ∂^2(uz)/∂(arg1)∂(arg2), ∂^2(uxx)/∂(arg1)∂(arg2), ..., ∂^2(uzz)/∂(arg1)∂(arg2)]` \
i.e.,
$$\left[\frac{\partial^2 u_x}{\partial\text{(arg1)}\partial\text{(arg2)}}, \frac{\partial^2 u_y}{\partial\text{(arg1)}\partial\text{(arg2)}}, \frac{\partial^2 u_z}{\partial\text{(arg1)}\partial\text{(arg2)}}, \frac{\partial^2}{\partial\text{(arg1)}\partial\text{(arg2)}}\left(\frac{\partial u_x}{\partial x}\right), \ldots, \frac{\partial^2}{\partial\text{(arg1)}\partial\text{(arg2)}}\left(\frac{\partial u_z}{\partial z}\right)\right].$$


If `False`, return is a list of 3 displacements differentiated by `arg`: \
`[∂^2(ux)/∂(arg1)∂(arg2), ∂^2(uy)/∂(arg1)∂(arg2), ∂^2(uz)/∂(arg1)∂(arg2)]`

The shape of each tensor is same as that of `x,y(,z)`.






### Examples


We have prepared a [notebook](../5_OkadaWrapper_hessian.ipynb) to test the `hessian` method.




---

- [Back to README.md](../README.md)
- [Go to the document of `SPOINT` and `SRECTF`](./Okada1985.md)
- [Go to the document of `DC3D0` and `DC3D`](./Okada1992.md)