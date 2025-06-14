# `OkadaWrapper` Class


`OkadaWrapper` is a convenient wrapper class to use functions `SPOINT`, `SRECTF`, `DC3D0` and `DC3D`.



The three methods of `OkadaWrapper`, `compute`, `gradient`, and `hessian`, all take as arguments the variables `coords` and `params` of type dict.
- `"coords"` is a dictionary that stores the coordinates of stations, allowing `"x", "y", "z"` as keys, and the corresponding values are tensors that represent the coordinate values.
- `"params"` is a dictionary that stores the values of fault parameters, allowing `"x_ref", "y_ref", "depth", "length", "width", "strike", "dip", "rake", "slip"` as keys.



The explanation of each key is as follows.
- `x,y,z`: Coordinates of the station. Values in the local Cartesian coordinate system with x in the east, y in the north, and z in the vertical. See also the `"Coordinate System"` section below. Note that z must be negative, and if you try to reference a point in air, the displacement/strain there will return 0.
- `x_ref, y_ref`: x,y coordinates of the epicenter. For a point source, these are the coordinates of the point, but for a rectangular fault, these are the coordinates of the top end of the fault (often called the "reference point"). It is not the coordinate of the center of the rectangle.
- `depth`: Depth of the source ($>0$). For a point source, this is the depth of the point, but for a rectangular fault, this is the coordinate of the top end of the fault ("reference point").
- `length, width`: These parameters are specific to a rectangular fault and, as the name implies, length (length of one side in the strike direction) and width (length of one side in the dip direction).
- `strike, dip, rake`: Specified in degrees.
- `slip`: This name can be misleading. For rectangular faults, this is as the name implies, the amount of slip. In the case of a point source, this represents the potency (seismic moment divided by the stiffness ratio, which is also equal to slip multiplied by the small area of the fault). Since it is complicated to specify arguments differently for point seismic sources and rectangular faults, they are specified with the same name in this way.

In addition, there are three optional arguments: `compute_strain`, `is_degree` and `nu`.
- `compute_strain`: The original FORTRAN subroutine computes both displacements and strains at a given location, but in some cases displacement alone may be sufficient. 
If `compute_strain` is `True` (default), both displacements and strains are computed.
If `False`, only the displacement is computed (In this cace, variables that are only used to compute strain are not used, thus reducing computational cost).
- `is_degree`: If `True`,  `"strike"`, `"dip"` and `"rake"` are in degrees. If `False`, they are in radians. 
- `nu`: Poisson's ratio of the assumed medium. Default value is 0.25, which means Poisson medium.







<!-- 単一観測点における変位と歪みを計算するには、次のようにします。


複数観測点における変位と歪みを計算する場合には、xとyを1次元または2次元のテンソルにするだけです。もちろん、xとyのdimとshapeは一致していなければなりません。 -->

Using a for-loop to calculate the outputs at multiple stations simultaneously would make the calculation very slow. 
For this reason, the `gradient` and `hessian` methods of `OkadaWrapper` use PyTorch's `vmap` function to parallelize the computation. 

> [!CAUTION]
> Since the author does not fully understand how `vmap` function works, it may exhibit unexpected behavior. 
> If you find such a thing, please report it to us.




> [!IMPORTANT]
> In the original subroutine, the coordinate system is set such that the x-axis is in the direction of strike (as defined in the paper), and this is carried over in these PyTorch implementations.
> `OkadaWrapper`, on the other hand, rotates the calculation results so that the values are in a local Cartesian coordinate system with x in the east, y in the north, and z in the vertically upward.




## `compute` method

Perform forward computations; given the source parameters, the displacements and/or their spatial derivatives at the station are calculated.

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


> [!IMPORTANT]
> In the `compute` method (and of cource, `gradient` and `hessian` method), the function to be called is determined by the keys of `coords` and `params`. That is,
> - if `x, y ∈ coords` but `z ∉ coords`, and `x_ref, y_ref, depth, strike, dip, rake, slip ∈ params` but `length, width ∉ params`, then `SPOINT` is called.
> - if `x, y ∈ coords` but `z ∉ coords`, and `x_ref, y_ref, depth, length, width, strike, dip, rake, slip ∈ params`, then `SRECTF` is called.
> - if `x, y, z ∈ coords`, and `x_ref, y_ref, depth, strike, dip, rake, slip ∈ params` but `length, width ∉ params`, then `DC3D0` is called.
> - if `x, y, z ∈ coords`, and `x_ref, y_ref, depth, length, width, strike, dip, rake, slip ∈ params`, then `DC3D` is called.
>
> If the required keys are missing, an error is raised. If unnecessary keys are included, they are ignored.

<!-- > - if `coords` has `"x", "y"` and not `"z"`, and `params` has `"x_ref", "y_ref", "depth", "strike", "dip", "rake", "slip"` and not `"length", "width"`, then `SPOINT` is called.
> - if `coords` has `"x", "y"` and not `"z"`, and `params` has `"x_ref", "y_ref", "depth", "length", "width", "strike", "dip", "rake", "slip"`, then `SRECTF` is called.
> - if `coords` has `"x", "y", "z"`, and `params` has `"x_ref", "y_ref", "depth", "strike", "dip", "rake", "slip"` and not `"length", "width"`, then `DC3D0` is called.
> - if `coords` has `"x", "y", "z"`, and `params` has `"x_ref", "y_ref", "depth", "length", "width", "strike", "dip", "rake", "slip"`, then `DC3D` is called. -->


### Outputs


If `compute_strain` is `True`, return is a list of 3 displacements and 9 spatial derivatives:
`[ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz]`

If `False`, return is a list of 3 displacements only:
`[ux, uy, uz]`

The shape of each tensor is same as that of `x,y(,z)`.

- `ux, uy, uz` : _torch.Tensor_
    - Displacement.
- `uxx, uyx, uzx` : _torch.Tensor_
    - x-derivative.
- `uxy, uyy, uzy` : _torch.Tensor_
    - y-derivative.
- `uxz, uyz, uzz` : _torch.Tensor_
    - z-derivative.


> [!NOTE]
> There's no `IRET` which existed in `DC3D0` and `DC3D`.

> [!NOTE]
> `uij` means $\frac{\partial u_i}{\partial x_j}$






    





## `gradient` method

Calculate gradient with respect to specified `arg` (one of coordinates or parameters) at the station, given the source parameters.
PyTorch's function `jacfwd` is used internally.

Currently, only a single `arg` can be specified.
If you want to get gradient with respect to multiple args, you need to call this method multiple times.



<!-- 
This method calculates the derivative of `f` with respect to spatial variables or with respect to parameters, using AD. 
PyTorch's function `jacfwd` is used internally.

If `"x", "y" (, "z")` is specified as `arg` (i.e., what is allowed as a key in the `coords`), the spatial derivative of `f` is computed. 
Of course, if there is no z in the `coords`, you cannot specify z as `arg`.
If `f` is displacement, the strain will be output. 
Since this is provided in the original Okada's formula, it is redundant to compute the strains with AD (simply using `compute` method is faster).
Note that it has been verified that the error is sufficiently small when the strain is calculated by the two methods. 
(See the corresponding Jupyter notebook.) 
If `f` is strain, it means that it is the second-order spatial derivative of the displacement, which cannot be computed with the original Okada's formula, so there is an advantage to computing it with AD.


If `"x_ref", "y_ref", "depth", "length", "width", "strike", "dip", "rake", "slip"` is specified as `arg` (i.e., what is allowed as a key in the `params`), the derivative of `f` with respect to parameters is calculated. 
Of course, if there is no length or width in `params`, you cannot specify length or width as `arg`. 
These are not provided in the original Okada's formula. 
You can also implement the derivatives with respect to the parameters by calculating them manually, but this is very time-consuming and may produce errors, so using AD is a better choice.

Note that if your task is to find parameters that minimize a certain loss function, it is not recommended to call `gradient` method explicitly. 
It is easier to use `loss.backward()`, as described later. -->





### Inputs
- `coords` : _dict of torch.Tensor_
    - same as that of `compute` method. 

- `params` : _dict of torch.Tensor_
    - same as that of `compute` method. 

- `arg` : _char_
    - Name of the variable to be differentiated. 
    This should be a key of `coords` or `params`.

- `compute_strain` : _bool, dafault True_
    - same as that of `compute` method. 

- `is_degree` : _bool, dafault True_
    - same as that of `compute` method. 

- `nu` : _float, default 0.25_
    - same as that of `compute` method. 







### Outputs



If `compute_strain` is `True`, return is a list of 3 displacements and 9 spatial derivatives differentiated by `arg`: \
`[∂(ux)/∂(arg), ∂(uy)/∂(arg), ∂(uz)/∂(arg), ∂(uxx)/∂(arg), ..., ∂(uzz)/∂(arg)]`

If `False`, return is a list of 3 displacements differentiated by `arg`: \
`[∂(ux)/∂(arg), ∂(uy)/∂(arg), ∂(uz)/∂(arg)]`

The shape of each tensor is same as that of `x,y(,z)`.



> [!TIP]
> `OkadaWrapper` can be used to find fault parameters that minimize a certain loss function (written in PyTorch function).
> In this case, the gradient value could be obtained explicitly by the `gradient` method and passed to the optimizer, but this would be redundant.
> Instead, it is easier to define a loss function, specify the parameters to be optimized, and then use `loss.backward()`.
> See the corresponding Jupyter notebook for more information on this.






## `hessian` method

Calculate hessian (2nd-order derivatives) with respect to specified `arg1` and `arg2` at the station, given the source parameters.



<!-- This method computes the second-order derivative of `f` with respect to spatial variables or with respect to parameters 
(again, `jacfwd` is used internally).
Theoretically, it is possible to differentiate `f` once by a spatial variable and once by a parameter.
However, this is not implemented for now.

If `"x", "y" (, "z")` is specified as `arg1` and `arg2` (i.e., what is allowed as a key in the `coords`), the second-order spatial derivative of `f` is computed. 
Of course, if there is no z in the `coords`, you cannot specify z as `arg1` or `arg2`.

If `"x_ref", "y_ref", "depth", "length", "width", "strike", "dip", "rake", "slip"` is specified as `arg1` and `arg2` (i.e., what is allowed as a key in the `params`), the second-order derivative of `f` with respect to parameters is calculated. 
Of course, if there is no length or width in `params`, you cannot specify length or width as `arg1` or `arg2`.  -->




### Inputs  
- `coords` : _dict of torch.Tensor_
    - same as that of `compute` method. 

- `params` : _dict of torch.Tensor_
    - same as that of `compute` method. 

- `arg1, arg2` : _char_
    - Name of the variable to be differentiated. 
    This should be a key of `coords` or `params`.
    Both `arg1` and `arg2` must be variables of the same kind; 
    both must be `coords` or both must be `params`.

- `compute_strain` : _bool, dafault True_
    - same as that of `compute` method. 

- `is_degree` : _bool, dafault True_
    - same as that of `compute` method. 

- `nu` : _float, default 0.25_
    - same as that of `compute` method. 

   




### Outputs



If `compute_strain` is `True`, return is a list of 3 displacements and 9 spatial derivatives differentiated by `arg1` and `arg2`: \
`[∂^2(ux)/∂(arg1)∂(arg2), ∂^2(uy)/∂(arg1)∂(arg2), ∂^2(uz)/∂(arg1)∂(arg2), ∂^2(uxx)/∂(arg1)∂(arg2), ..., ∂^2(uzz)/∂(arg1)∂(arg2)]`

If `False`, return is a list of 3 displacements differentiated by `arg`: \
`[∂^2(ux)/∂(arg1)∂(arg2), ∂^2(uy)/∂(arg1)∂(arg2), ∂^2(uz)/∂(arg1)∂(arg2)]`

The shape of each tensor is same as that of `x,y(,z)`.