# How to use `DC3D0` and `DC3D`



## `DC3D0`


Calculate displacement and strain at depth due to buried point source in a semiinfinite medium.

### Inputs

- `ALPHA` : _float or torch.Tensor_
    - Medium constant. $\frac{\lambda+\mu}{\lambda+2\mu}$
- `X, Y, Z` : _torch.Tensor_
    - Coordinate of observing point.
- `DEPTH` : _float or torch.Tensor_
    - Source depth.
- `DIP` : _torch.Tensor_
    - Dip-angle.
- `POT1, POT2, POT3, POT4` : _float or torch.Tensor_
    - Strike-, dip-, tensile- and inflate-potency.
    - potency = (moment of double-couple)/myu for POT1,2
    - potency = (intensity of isotropic part)/lambda for POT3
    - potency = (intensity of linear dipole)/myu for POT4
- `compute_strain` : _bool, dafault True_
    - Option to calculate the spatial derivative of the displacement. 
    New in the PyTorch implementation.
- `is_degree` : _bool, dafault True_
    - Flag if `DIP` is in degree or not (= in radian). 
    New in the PyTorch implementation.


### Outputs

- U : _list of torch.Tensor_
    - If `compute_strain` is `True`, return is a list of 3 displacements and 9 spatial derivatives:
    `[UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]`
    - If `False`, return is a list of 3 displacements only:
    `[UX, UY, UZ]`

    - `UX, UY, UZ` : _torch.Tensor_
        - Displacement. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**2
    - `UXX, UYX, UZX` : _torch.Tensor_
        - X-derivative. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**3
    - `UXY, UYY, UZY` : _torch.Tensor_
        - Y-derivative. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**3
    - `UXZ, UYZ, UZZ` : _torch.Tensor_
        - Z-derivative. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**3

- IRET : _torch.Tensor (int)_
    - Return code.
    - `IRET=0` means normal, `IRET=1` means singular, `IRET=2` means positive z was given.



> [!NOTE]
> `Uij` means $\frac{\partial U_i}{\partial x_j}$.


<!-- ### Examples 

単一観測点における変位と歪みを計算するには、次のようにします。


複数観測点における変位と歪みを計算する場合には、xとyを1次元または2次元または3次元のテンソルにするだけです。もちろん、xとyのdimとshapeは一致していなければなりません。



-->








## `DC3D`

Calculate displacement and strain at depth due to buried finite fault in a semiinfinite medium.

### Inputs

- `ALPHA` : _float or torch.Tensor_
    - Medium constant. $\frac{\lambda+\mu}{\lambda+2\mu}$
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


### Outputs

- U : _list of torch.Tensor_
    - If `compute_strain` is `True`, return is a list of 3 displacements and 9 spatial derivatives:
    `[UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]`
    - If `False`, return is a list of 3 displacements only:
    `[UX, UY, UZ]`

    - `UX, UY, UZ` : _torch.Tensor_
        - Displacement. unit = (unit of dislocation)
    - `UXX, UYX, UZX` : _torch.Tensor_
        - X-derivative. unit = (dislocation) / (unit of X,Y,Z,DEPTH,AL,AW)
    - `UXY, UYY, UZY` : _torch.Tensor_
        - Y-derivative. unit = (dislocation) / (unit of X,Y,Z,DEPTH,AL,AW)
    - `UXZ, UYZ, UZZ` : _torch.Tensor_
        - Z-derivative. unit = (dislocation) / (unit of X,Y,Z,DEPTH,AL,AW)

- IRET : _torch.Tensor (int)_
    - Return code.
    - `IRET=0` means normal, `IRET=1` means singular, `IRET=2` means positive z was given.



<!-- ### Examples 

単一観測点における変位と歪みを計算するには、次のようにします。


複数観測点における変位と歪みを計算する場合には、xとyを1次元または2次元または3次元のテンソルにするだけです。もちろん、xとyのdimとshapeは一致していなければなりません。


-->




<!-- > [!NOTE]

If you are familiar with the `dc3d0wrapper` or `dc3dwrapper` from [okada_wrapper](https://github.com/cutde-org/okada_wrapper), you can use similar interfaces by defining the following functions.
> ```python
> import torch
> from OkadaTorch import DC3D0, DC3D
> 
> def dc3d0wrapper(alpha, xo, depth, dip, potency):
>     u = torch.empty(3)
>     grad_u = torch.empty((3, 3))
>     u[0], u[1], u[2], \
>     grad_u[0, 0], grad_u[0, 1], grad_u[0, 2], \
>     grad_u[1, 0], grad_u[1, 1], grad_u[1, 2], \
>     grad_u[2, 0], grad_u[2, 1], grad_u[2, 2] = \
>     DC3D0(alpha, xo[0], xo[1], xo[2], depth, dip, 
>           potency[0], potency[1], potency[2], potency[3], True, True)
>     return u, grad_u
> 
> def dc3dwrapper(alpha, xo, depth, dip, strike_width, dip_width, dislocation):
>     u = torch.empty(3)
>     grad_u = torch.empty((3, 3))
>     u[0], u[1], u[2], \
>     grad_u[0, 0], grad_u[0, 1], grad_u[0, 2], \
>     grad_u[1, 0], grad_u[1, 1], grad_u[1, 2], \
>     grad_u[2, 0], grad_u[2, 1], grad_u[2, 2] = \
>     DC3D(alpha, xo[0], xo[1], xo[2], depth, dip, 
>          strike_width[0], strike_width[1], dip_width[0], dip_width[1],
>          dislocation[0], dislocation[1], dislocation[2], True, True)
>     return u, grad_u
> ```


> [!TIP]
> This method requires the use of for-loop to obtain displacements and strains at multiple stations. Our `OkadaWrapper.compute` uses vectorization to obtain results at multiple stations without using for-loop. -->