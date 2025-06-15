# How to use `OkadaWrapper` Class

`OkadaWrapper` is a convenient wrapper class that can handle `SPOINT`, `SRECTF`, `DC3D0` and `DC3D` with common interface.
It also provides functions to calculate gradient and hessian.


## Introduction


The `OkadaWrapper` class has three methods (`compute`, `gradient`, and `hessian`). 
Here, their common arguments, `coords`, `params`, `compute_strain`, `is_degree`, and `nu`, are explained first.


### `coords`

`coords` is a Python dictionary that stores the coordinates of stations.
Allowed keys are `"x"`, `"y"`, `"z"` and the corresponding values are `torch.Tensor`s representing the coordinates of the stations. 
In this coordinate system, x is east, y is north, and z is upward. Note that the values of z must be negative.



### `params`

`params` is also a Python dictionary that stores the values of source parameters.
Allowed keys are `"x_ref"`, `"y_ref"`, `"depth"`, `"length"`, `"width"`, `"strike"`, `"dip"`, `"rake"`, `"slip"`.
The explanation of each key is as follows.

- `x_ref, y_ref`: x, y coordinates of the source. 
For a point source, these are the coordinates of the point.
For a rectangular fault, these are the coordinates of the top left corner of the rectangle (often refered as "reference point"), not the center of the rectangle.
- `depth`: Depth of the source ($>0$). 
For a point source, this is the depth of the point.
For a rectangular fault, this is the coordinate of the reference point.
- `length, width`: These parameters are specific to a rectangular fault.
- `strike, dip, rake`: If `is_degree` is `True`, these variables are measured in degrees. If `False`, in radians. 
- `slip`: This name can be misleading. 
For a rectangular fault, this indicates the amount of slip, as the name implies.
For a point source, this indicates the potency (which is equal to seismic moment divided by the elastic constant, also equal to slip amount multiplied by the infinitesimal area of the fault). 
For ease of implementation, they are treated under the same name `slip`.


### `compute_strain`
The original FORTRAN subroutine computes both displacements and strains, but in some cases displacement alone may be sufficient. 
If `compute_strain` is `True` (default), both displacements and strains are computed.
If `False`, only the displacement is computed.
In this cace, intermediate variables that are only used to compute strain are not assigned, thus reducing computational cost.


### `is_degree`
If `True` (default),  `"strike"`, `"dip"` and `"rake"` are in degrees. If `False`, in radians. 


### `nu`
Poisson's ratio of the assumed medium. Default value is 0.25, which means Poisson medium.








## `compute` method

Perform forward computations; given the source parameters, the displacements and/or their spatial derivatives at the stations are calculated.

Currently, multiple station coordinates can be specified, but only one set of source parameters can be specified. 
If you have multiple sources, you need to call this method multiple times.


### Inputs

- `coords` : _dict of torch.Tensor_
    - `"x"` and `"y"` are required keys, and `"z"` is optional (all other keys are ignored).
    Each value must be torch.Tensor of the same shape (`dim` is arbitrary).

- `params` : _dict of torch.Tensor_
    - `"x_ref"`, `"y_ref"`, `"depth"`, `"strike"`, `"dip"`, `"rake"` and `"slip"` are required keys, and `"length"` and `"width"` are optional (all other keys are ignored).
    Each value must be torch.Tensor with dim=0 (scaler tensor).

- `compute_strain` : _bool, dafault True_
    - Option to calculate the spatial derivative of the displacement.

- `is_degree` : _bool, dafault True_
    - Flag if `"strike"`, `"dip"` and `"rake"` are in degree or not (= in radian). 

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
> In the `compute` method (and of cource, `gradient` and `hessian` method), the function to be called is determined by the keys of `coords` and `params`. That is,
> - if `x, y ∈ coords` but `z ∉ coords`, and `x_ref, y_ref, depth, strike, dip, rake, slip ∈ params` but `length, width ∉ params`, then `SPOINT` is called.
> - if `x, y ∈ coords` but `z ∉ coords`, and `x_ref, y_ref, depth, length, width, strike, dip, rake, slip ∈ params`, then `SRECTF` is called.
> - if `x, y, z ∈ coords`, and `x_ref, y_ref, depth, strike, dip, rake, slip ∈ params` but `length, width ∉ params`, then `DC3D0` is called.
> - if `x, y, z ∈ coords`, and `x_ref, y_ref, depth, length, width, strike, dip, rake, slip ∈ params`, then `DC3D` is called.
>
> If the required keys are missing, an error is raised. If unnecessary keys are included, they are ignored.





> [!NOTE]
> There's no `IRET` which existed in `DC3D0` and `DC3D`.





### Examples

<!-- 単一観測点における変位と歪みを計算するには、次のようにします。


複数観測点における変位と歪みを計算する場合には、xとyを1次元または2次元のテンソルにするだけです。もちろん、xとyのdimとshapeは一致していなければなりません。 -->
    





## `gradient` method

Calculate gradient with respect to specified `arg` (one of coordinates or parameters) at the stations, given the source parameters.
PyTorch's function `jacfwd` is used internally.

> [!NOTE]
> Only a single `arg` can be specified.
> If you want to get gradient with respect to multiple args, you need to call this method multiple times.





If `"x", "y" (, "z")` is specified as `arg` (i.e., what is allowed as a key in the `coords`), the spatial derivative of `u` is calculated. 
If the component of `u` is displacement, the strain will be output. 
Since this is provided in the original Okada's formula, it is redundant to compute the strains with AD (simply using `compute` method is faster).
Note that it has been verified that the error is sufficiently small when the strain is calculated by the two methods. 
(See the corresponding Jupyter notebook.) 
If the component of `u` is strain, it means that it is the second-order spatial derivative of the displacement, which cannot be computed with the original Okada's formula, so there is an advantage to computing it with AD.


If `"x_ref", "y_ref", "depth", "length", "width", "strike", "dip", "rake", "slip"` is specified as `arg` (i.e., what is allowed as a key in the `params`), the derivative of `u` with respect to parameters is calculated. 
These are not provided in the original Okada's formula. 
You can implement the derivatives with respect to the parameters by calculating them manually, but this is very time-consuming and may produce errors, so using AD is a better choice.







### Inputs
- `coords` : _dict of torch.Tensor_
    - same as that of `compute` method. 

- `params` : _dict of torch.Tensor_
    - same as that of `compute` method. 

- `arg` : _char_
    - Name of the variable to be differentiated. 
    This should be a key of `coords` or `params`; 
    if there is no `"z"` in `coords`, you cannot specify `"z"` as `arg`.
    Similarly, if there is no `"length"` or `"width"` in `params`, you cannot specify `"length"` or `"width"` as `arg`. 

- `compute_strain` : _bool, dafault True_
    - same as that of `compute` method. 

- `is_degree` : _bool, dafault True_
    - same as that of `compute` method. 

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
> See the corresponding Jupyter notebook for more information on this.




### Examples







## `hessian` method

Calculate hessian (2nd-order derivatives) with respect to specified `arg1` and `arg2` at the station, given the source parameters.
PyTorch's function `jacfwd` is used internally.

Theoretically, it is possible to differentiate `u` once by a spatial variable and once by a parameter.However, this is not implemented.
**Both `arg1` and `arg2` must be variables of the same kind; both must be `coords` or both must be `params`.**

If `"x", "y" (, "z")` is specified as `arg1` and `arg2` (i.e., what is allowed as a key in the `coords`), the second-order spatial derivative of `u` is calculated. 

If `"x_ref", "y_ref", "depth", "length", "width", "strike", "dip", "rake", "slip"` is specified as `arg1` and `arg2` (i.e., what is allowed as a key in the `params`), the second-order derivative of `u` with respect to parameters is calculated. 





### Inputs  
- `coords` : _dict of torch.Tensor_
    - same as that of `compute` method. 

- `params` : _dict of torch.Tensor_
    - same as that of `compute` method. 

- `arg1, arg2` : _char_
    - Name of the variable to be differentiated. 
    This should be a key of `coords` or `params`;
    if there is no `"z"` in `coords`, you cannot specify `"z"` as `arg1` or `arg2`.
    Similarly, if there is no `"length"` or `"width"` in `params`, you cannot specify `"length"` or `"width"` as `arg1` or `arg2`. 
    

- `compute_strain` : _bool, dafault True_
    - same as that of `compute` method. 

- `is_degree` : _bool, dafault True_
    - same as that of `compute` method. 

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






---

- [Back to README.md](../README.md)
- [Go to "How to use `SPOINT` and `SRECTF`"](./Okada1985.md)
- [Go to "How to use `DC3D0` and `DC3D`"](./Okada1992.md)