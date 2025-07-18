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
In this coordinate system, x is east, y is north, and z is upward. Note that the values of z must be negative.



### `params`

`params` is also a Python dictionary that stores the values of source parameters.
Allowed keys are `"x_fault"`, `"y_fault"`, `"depth"`, `"length"`, `"width"`, `"strike"`, `"dip"`, `"rake"`, `"slip"`.
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


### `compute_strain`
The original FORTRAN subroutine computes both displacements and strains, but in some cases displacement alone may be sufficient. 
If `compute_strain` is `True` (default), both displacements and strains are computed.
If `False`, only the displacement is computed.
In this case, intermediate variables that are only used to compute strain are not assigned, thus reducing computational cost.


### `is_degree`
If `True` (default),  `"strike"`, `"dip"` and `"rake"` are in degrees. If `False`, in radians. 


### `fault_origin`
In the case of a rectangular fault, this flag specifies which point the fault location parameter refers to (ignored for a point source).
- If `fault_origin` is `"topleft"`, then `"x_fault"`, `"y_fault"` and `"depth"` in `params` represent the coordinates of the top left corner of the rectangle.
- If `fault_origin` is `"center"`, then `"x_fault"`, `"y_fault"` and `"depth"` in `params` represent the coordinates of the rectangle's center.

Other strings cannot be specified. 


### `nu`
Poisson's ratio of the assumed medium. Default value is 0.25, which means Poisson medium.








## `OkadaWrapper.compute`(_coords:dict, params:dict, compute_strain:bool=True, is_degree:bool=True, fault_origin:str="topleft", nu:float=0.25_)

Perform forward computations; given the source parameters, the displacements and/or their spatial derivatives at the stations are calculated.

Currently, multiple station coordinates can be specified, but only one set of source parameters can be specified. 
If you have multiple sources, you need to call this method multiple times.


### Inputs

- `coords` : _dict of torch.Tensor_
    - `"x"` and `"y"` are required keys, and `"z"` is optional (all other keys are ignored).
    Each value must be torch.Tensor of the same shape (`dim` is arbitrary).

- `params` : _dict of torch.Tensor_
    - `"x_fault"`, `"y_fault"`, `"depth"`, `"strike"`, `"dip"`, `"rake"` and `"slip"` are required keys, and `"length"` and `"width"` are optional (all other keys are ignored).
    Each value must be torch.Tensor with dim=0 (scaler tensor).

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

<!-- outputの単位については、呼び出されているそれぞれの関数の説明を見てください。 -->

> [!IMPORTANT]
> In the `compute` method (and of course, `gradient` and `hessian` method), the function to be called is determined by the keys of `coords` and `params`. That is,
> - if `x, y ∈ coords` but `z ∉ coords`, and `x_fault, y_fault, depth, strike, dip, rake, slip ∈ params` but `length, width ∉ params`, then `SPOINT` is called.
> - if `x, y ∈ coords` but `z ∉ coords`, and `x_fault, y_fault, depth, length, width, strike, dip, rake, slip ∈ params`, then `SRECTF` is called.
> - if `x, y, z ∈ coords`, and `x_fault, y_fault, depth, strike, dip, rake, slip ∈ params` but `length, width ∉ params`, then `DC3D0` is called.
> - if `x, y, z ∈ coords`, and `x_fault, y_fault, depth, length, width, strike, dip, rake, slip ∈ params`, then `DC3D` is called.
>
> If the required keys are missing, an error is raised. If unnecessary keys are included, they are ignored.





> [!NOTE]
> There's no `IRET` which existed in `DC3D0` and `DC3D`.





### Examples

We have prepared a [notebook](../3_OkadaWrapper_compute.ipynb) to test the `compute` method.


> [!NOTE]
> Source parameters used in the notebooks were taken from the model 10 of Table S1 in Baba et al. 2021.
> - Baba, T., Chikasada, N., Imai, K., Tanioka, Y., & Kodaira, S., 2021. 
Frequency dispersion amplifies tsunamis caused by outer-rise normal faults, Scientific Reports, 11(1), 20064, 
doi: https://doi.org/10.1038/s41598-021-99536-x.




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


If `"x_fault", "y_fault", "depth", "length", "width", "strike", "dip", "rake", "slip"` is specified as `arg` (i.e., what is allowed as a key in the `params`), the derivative of `u` with respect to parameters is calculated. 
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
    Similarly, if there is no `"length"` or `"width"` in `params`, you cannot specify `"length"` or `"width"` as `arg`. 

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

If `"x_fault", "y_fault", "depth", "length", "width", "strike", "dip", "rake", "slip"` is specified as `arg1` and `arg2` (i.e., what is allowed as a key in the `params`), the second-order derivative of `u` with respect to parameters is calculated. 





### Inputs  
- `coords` : _dict of torch.Tensor_
    - same as that of `compute` method. 

- `params` : _dict of torch.Tensor_
    - same as that of `compute` method. 

- `arg1, arg2` : _str_
    - Name of the variable to be differentiated. 
    This should be a key of `coords` or `params`;
    if there is no `"z"` in `coords`, you cannot specify `"z"` as `arg1` or `arg2`.
    Similarly, if there is no `"length"` or `"width"` in `params`, you cannot specify `"length"` or `"width"` as `arg1` or `arg2`. 
    

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