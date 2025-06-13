# How to use `DC3D0` and `DC3D`


> [!NOTE]
> In the following, `Uij` means $\frac{\partial U_i}{\partial x_j}$.



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

If `compute_strain` is `True`, return is a list of 3 displacements and 9 spatial derivatives:
`[UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]`

If `False`, return is a list of 3 displacements only:
`[UX, UY, UZ]`

- `UX, UY, UZ` : _torch.Tensor_
    - Displacement. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**2
- `UXX, UYX, UZX` : _torch.Tensor_
    - X-derivative. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**3
- `UXY, UYY, UZY` : _torch.Tensor_
    - Y-derivative. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**3
- `UXZ, UYZ, UZZ` : _torch.Tensor_
    - Z-derivative. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**3




<!-- ### Examples -->






---

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
If `compute_strain` is `True`, return is a list of 3 displacements and 9 spatial derivatives:
`[UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]`

If `False`, return is a list of 3 displacements only:
`[UX, UY, UZ]`

- `UX, UY, UZ` : _torch.Tensor_
    - Displacement. unit = (unit of dislocation)
- `UXX, UYX, UZX` : _torch.Tensor_
    - X-derivative. unit = (dislocation) / (unit of X,Y,Z,DEPTH,AL,AW)
- `UXY, UYY, UZY` : _torch.Tensor_
    - Y-derivative. unit = (dislocation) / (unit of X,Y,Z,DEPTH,AL,AW)
- `UXZ, UYZ, UZZ` : _torch.Tensor_
    - Z-derivative. unit = (dislocation) / (unit of X,Y,Z,DEPTH,AL,AW)


<!-- ### Examples -->
