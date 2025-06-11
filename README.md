# OkadaTorch

Port code of Okada (1985, 1992) written in PyTorch.

References:
- Okada, Y. (1985). Surface deformation due to shear and tensile faults in a half-space. Bulletin of the seismological society of America, 75(4), 1135-1154.
https://doi.org/10.1785/BSSA0750041135
- Okada, Y. (1992). Internal deformation due to shear and tensile faults in a half-space. Bulletin of the seismological society of America, 82(2), 1018-1040.
https://doi.org/10.1785/BSSA0820021018



**以下はまだ書きかけなので信用しないでください**



## Install

```
git clone https://github.com/msomeya1/OkadaTorch
```

## Basic Usage


Okada (1985, 1992) provides four subroutines:
- SPOINT (Okada 1985): displacements and strains (spatial derivative of displacements) at surface ($z=0$) created by a point source.
- SRECTF (Okada 1985): displacements and strains at surface ($z=0$) created by a rectangular fault.
- DC3D0 (Okada 1992): Same as SPOINT, but in depth ($z\leq0$).
- DC3D (Okada 1992): Same as SRECTF, but in depth ($z\leq0$).


||At Surface (Okada 1985)|In Depth (Okada 1992)|
|-|-|-|
|Point Source|SPOINT|SRECTF|
|Rectangular Fault|DC3D0|DC3D|

This directory `OkadaTorch` provides PyTorch implementations of these four subroutines. 
The inputs and outputs are almost identical to the original FORTRAN subroutine, but there is a new option called `compute_strain`.
The original FORTRAN subroutine computes both displacements and strains at a given location, but in some cases displacement alone may be sufficient. 
If `compute_strain` is `False`, only the displacement is computed (In this cace, variables that are only used to compute strain are not used, thus reducing computational cost).




## Wrapper Function

Although we could have ended there, we have prepared wrapper functions to make these subroutines convenient to use. 
We will explain them below.


The three methods of `OkadaWrapper`, `compute`, `gradient`, and `hessian`, all take as arguments the variables `coords` and `params` of type dict.
As the name suggests, `"coords"` is a dictionary that stores the coordinates of observation points, allowing `"x", "y", "z"` as keys, and the corresponding values are tensors that represent the coordinate values.
`"params"` is a dictionary that stores the values of fault parameters, allowing `"x_ref", "y_ref", "depth", "length", "width", "strike", "dip", "rake", "slip"` as keys.


The explanation of each key is as follows.
- `"x,y,z"`: Coordinates of the observation point. Values in the local Cartesian coordinate system with x in the east, y in the north, and z in the vertical. See also the `"Coordinate System"` section below. Note that z must be negative, and if you try to reference a point in air, the displacement/strain there will return 0.
- `"x_ref, y_ref"`: x,y coordinates of the epicenter. For a point source, these are the coordinates of the point, but for a rectangular fault, these are the coordinates of the top end of the fault (often called the "reference point"). It is not the coordinate of the center of the rectangle.
- `"depth"`: Depth of the source ($>0$). For a point source, this is the depth of the point, but for a rectangular fault, this is the coordinate of the top end of the fault ("reference point").
- `"length, width"`: These parameters are specific to a rectangular fault and, as the name implies, length (length of one side in the strike direction) and width (length of one side in the dip direction).
- `"strike, dip, rake"`: Specified in degrees.
- `"slip"`: This name can be misleading. For rectangular faults, this is as the name implies, the amount of slip. In the case of a point source, this represents the potency (seismic moment divided by the stiffness ratio, which is also equal to slip multiplied by the small area of the fault). Since it is complicated to specify arguments differently for point seismic sources and rectangular faults, they are specified with the same name in this way.

In addition, there is an optional argument `nu`. This is the Poisson's ratio of the assumed medium, which is 0.25 by default (i.e., Poisson medium).

The other arguments, `compute_strain` and `arg`, are explained separately later.




### Remark 1
In this wrapper function, the function to be called is controlled by what the keys of `coords` and `params` are. That is,
- if (1) `coords` contains `"x", "y"` and not `"z"`, and (2) `params` contains `"x_ref", "y_ref", "depth", "strike", "dip", "rake", "slip"` and not `"length", "width"`, then `SPOINT` is called.
- if (1) `coords` contains `"x", "y"` and not `"z"`, and (2) `params` contains `"x_ref", "y_ref", "depth", "length", "width", "strike", "dip", "rake", "slip"`, then `SRECTF` is called.
- if (1) `coords` contains `"x", "y", "z"`, and (2) `params` contains `"x_ref", "y_ref", "depth", "strike", "dip", "rake", "slip"` and not `"length", "width"`, then `DC3D0` is called.
- if (1) `coords` contains `"x", "y", "z"`, and (2) `params` contains `"x_ref", "y_ref", "depth", "length", "width", "strike", "dip", "rake", "slip"`, then `DC3D` is called.

If the required keys are missing, an error is raised. If unnecessary keys are included, they are ignored.


### Remark 2
The most important feature of this wrapper function is that
- there can be multiple observation points
- but only one set of fault parameters (parameters for one fault) is allowed. 

In other words, it is not possible to have multiple faults and calculate the displacements, etc. (if you want to do such a thing, simply call `OkadaWrapper` multiple times).

For the purposes described above, each value of `params` must be a scalar tensor.
On the other hand, `"x", "y" (, "z")` can be a 1D or 2D or 3D tensor (of course, these shapes must be common). 
In this case, the value of the output at each observation point is output as a tensor of the same shape as `"x", "y" (, "z")`.

Using a for-loop to calculate the outputs at multiple observation points simultaneously would make the calculation very slow. 
For this reason, the `gradient` and `hessian` methods of `OkadaWrapper` use PyTorch's `vmap` function to parallelize the computation. 
However, since the author does not fully understand how `vmap` function works, it may exhibit unexpected behavior. 
If you find such a thing, please report it to us.





## Returns

### `compute` Method
This method simply performs a forward calculation. 
That is, under the specified observation points and fault parameters, displacements and strains (spatial derivatives of displacements) are output (only displacements when `compute_strain` is `False`).

For the following explanation, any of the outputs (3 displacement and 9 strains) will be denoted as `f`.


### `gradient` Method
This method calculates the derivative of `f` with respect to spatial variables or with respect to parameters, using automatic differentiation (AD). 
PyTorch's function `jacfwd` is used internally.

If `"x", "y" (, "z")` is specified as `arg` (i.e., what is allowed as a key in the `coords`), the spatial derivative of `f` is computed. 
Of course, if there is no z in the `coords`, you cannot specify z as `arg`.
If `f` is displacement, the strain will be output. 
Since this is provided in the original Okada's formula, it is redundant to compute the strains with AD (simply using `compute` method is faster).
Note that it has been verified that the error is sufficiently small when the strain is calculated by the two methods. 
(See the corresponding Jupyter notebook.) 
If `f` is strain, it means that it is the second-order spatial derivative of the displacement, which cannot be computed with the original Okada's formula, so there is an advantage to computing it with AD.

If `compute_strain` is `True` and `arg` is x, the output is ... (to be written)

If `"x_ref", "y_ref", "depth", "length", "width", "strike", "dip", "rake", "slip"` is specified as `arg` (i.e., what is allowed as a key in the `params`), the derivative of `f` with respect to parameters is calculated. 
Of course, if there is no length or width in `params`, you cannot specify length or width as `arg`. 
These are not provided in the original Okada's formula. 
You can also implement the derivatives with respect to the parameters by calculating them manually, but this is very time-consuming and may produce errors, so using AD is a better choice.

Note that if your task is to find parameters that minimize a certain loss function, it is not recommended to call `gradient` method explicitly. 
It is easier to use `loss.backward()`, as described later.


### `hessian` Method
This method computes the second-order derivative of `f` with respect to spatial variables or with respect to parameters 
(again, `jacfwd` is used internally).
Theoretically, it is possible to differentiate `f` once by a spatial variable and once by a parameter.
However, this is not implemented for now.

If `"x", "y" (, "z")` is specified as `arg1` and `arg2` (i.e., what is allowed as a key in the `coords`), the second-order spatial derivative of `f` is computed. 
Of course, if there is no z in the `coords`, you cannot specify z as `arg1` or `arg2`.

If `"x_ref", "y_ref", "depth", "length", "width", "strike", "dip", "rake", "slip"` is specified as `arg1` and `arg2` (i.e., what is allowed as a key in the `params`), the second-order derivative of `f` with respect to parameters is calculated. 
Of course, if there is no length or width in `params`, you cannot specify length or width as `arg1` or `arg2`. 





## Parameter Estimation using `loss.backward()`

`OkadaWrapper` can be used to find fault parameters that minimize a certain loss function (written in PyTorch function).
In this case, the gradient value could be obtained explicitly by the `gradient` method and passed to the optimizer, but this would be redundant.
Instead, it is easier to define a loss function, specify the parameters to be optimized, and then use `loss.backward()`.
See the corresponding Jupyter notebook for more information on this.



## Coordinate System
In the original subroutine, the coordinate system is set such that the x-axis is in the direction of strike (as defined in the paper), and this is carried over in these PyTorch implementations.
The wrapper function, on the other hand, rotates the calculation results so that the values are in a local Cartesian coordinate system with x in the east, y in the north, and z in the vertically upward.







