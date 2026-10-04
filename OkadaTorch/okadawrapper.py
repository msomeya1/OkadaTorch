import functools
import warnings

import torch
from torch.func import jacfwd, vmap
from .okada1985 import SPOINT, SRECTF
from .okada1992 import DC3D0, DC3D
from .geometry import setup, rotate_vector, rotate_tensor


COORDINATE_AXIS = {"x": 0, "y": 1, "z": 2}   # position of each coordinate in _fn

REQUIRED_COORDS  = ("x", "y")
OPTIONAL_COORDS  = ("z",)
REQUIRED_PARAMS  = ("x_fault", "y_fault", "depth", "strike", "dip", "rake", "slip")
RECTANGLE_PARAMS = ("length", "width")
# Third and fourth source components; the kernels always supported them, only
# the wrapper hard-coded them to zero.
OPTIONAL_PARAMS  = ("opening", "inflation")
FAULT_ORIGINS    = ("topleft", "center")

KNOWN_COORDS = REQUIRED_COORDS + OPTIONAL_COORDS
KNOWN_PARAMS = REQUIRED_PARAMS + RECTANGLE_PARAMS + OPTIONAL_PARAMS

# Bounded by the requirement that the elastic energy be positive definite.
# nu = 0.5 is the incompressible limit, where the 1985 medium constant 1 - 2*nu
# vanishes; the formulae stay finite there, so it would not fail on its own.
NU_RANGE = (-1.0, 0.5)


def _common_dtype(tensors):
    """The dtype every input is cast to, using torch's own promotion rule.

    `torch.from_numpy` gives float64 while `torch.tensor(30.0)` gives float32, so
    a mixture is what the ordinary numpy-to-torch idiom produces.  Promoting
    cannot lose precision; it was demotion -- the old behaviour, where `coords`
    decided for everyone -- that silently did.
    """
    if not tensors:
        return torch.get_default_dtype()
    return functools.reduce(torch.promote_types, (t.dtype for t in tensors))


def _validate(coords, params, fault_origin="topleft", nu=0.25):
    """Check the inputs, normalise them, and report which kernel applies.

    Returns `(coords, params, is_rectangular)` with recognised Python scalars
    promoted to tensors -- `torch.deg2rad` rejects plain floats, which used to
    make `strike=30.0` a TypeError from deep inside the kernel.

    Raises `ValueError` rather than using `assert`, which `python -O` removes.
    """
    # Unrecognised keys are rejected: the previous contract -- ignore them --
    # turned 'Z' into a silent switch to the surface formulation and 'widht'
    # into a silent switch to a point source.
    for label, mapping, known in (("coords", coords, KNOWN_COORDS),
                                  ("params", params, KNOWN_PARAMS)):
        unknown = sorted(k for k in mapping if k not in known)
        if unknown:
            raise ValueError(
                f"'{label}' has unrecognized keys {unknown}. Allowed keys are "
                f"{list(known)}.")

    missing = [k for k in REQUIRED_COORDS if k not in coords]
    if missing:
        raise ValueError(
            f"'coords' is missing {missing}. Required keys are "
            f"{list(REQUIRED_COORDS)}; 'z' is optional and selects the "
            f"Okada (1992) formulation.")

    missing = [k for k in REQUIRED_PARAMS if k not in params]
    if missing:
        raise ValueError(
            f"'params' is missing {missing}. Required keys are "
            f"{list(REQUIRED_PARAMS)}.")

    supplied = [v for mapping, known in ((coords, KNOWN_COORDS),
                                        (params, KNOWN_PARAMS))
                for k, v in mapping.items() if k in known and torch.is_tensor(v)]

    # One device throughout -- unlike dtypes, these cannot be reconciled without
    # guessing which one the caller meant.  Mixing was accepted silently, and a
    # 0-dim CPU parameter reaches a CUDA kernel as a scalar, which rounds
    # differently from the same value held on the device.
    devices = {v.device for v in supplied}
    if len(devices) > 1:
        raise ValueError(
            f"all inputs must be on one device, got "
            f"{sorted(str(d) for d in devices)}.")
    device = devices.pop() if devices else torch.device("cpu")

    used_dtype = _common_dtype(supplied)

    def _cast(mapping, known):
        def one(v):
            if torch.is_tensor(v):
                # `.to` keeps the autograd graph; `torch.as_tensor` is only for
                # the plain Python numbers, which have none.
                return v if v.dtype == used_dtype else v.to(used_dtype)
            return torch.as_tensor(v, dtype=used_dtype, device=device)
        return {k: (one(v) if k in known else v) for k, v in mapping.items()}

    coords = _cast(coords, KNOWN_COORDS)
    params = _cast(params, KNOWN_PARAMS)

    if used_dtype in (torch.float32, torch.float16, torch.bfloat16):
        warnings.warn(
            f"OkadaTorch is running in {used_dtype}. On a large fault this "
            f"costs up to around 0.2% on the displacements and leaves the gradients "
            f"good to three or four digits, which is usually fine alongside a "
            f"neural network, but a small fault far from the stations can be "
            f"off by a few percent. Prefer float64 for such a fault, for one that is shallow "
            f"compared with its own dimensions, or when the gradients "
            f"themselves are the result.",
            stacklevel=3)

    shapes = {k: tuple(coords[k].shape) for k in ("x", "y", "z") if k in coords}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"all coordinates must have the same shape, got {shapes}.")

    # A rectangle needs both of length/width and a point source needs neither.
    # Supplying exactly one used to fall through to the point-source branch
    # without a word, so a typo such as 'widht' silently changed the model.
    given = [k for k in RECTANGLE_PARAMS if k in params]
    if len(given) == 1:
        absent = [k for k in RECTANGLE_PARAMS if k not in params]
        raise ValueError(
            f"'params' contains {given} but not {absent}. A rectangular fault "
            f"needs both of {list(RECTANGLE_PARAMS)}; a point source needs "
            f"neither. Supplying only one is ambiguous.")

    if "inflation" in params:
        if len(given) == 2:
            raise ValueError(
                "'inflation' is the isotropic (volume-change) component of a "
                "point source; the rectangular kernels have no such term. Drop "
                "'inflation', or drop 'length'/'width' to model a point source.")
        if "z" not in coords:
            raise ValueError(
                "'inflation' requires 'z' in 'coords'. The Okada (1985) surface "
                "routine SPOINT has only three source components (strike-slip, "
                "dip-slip, tensile); the isotropic one exists only in the (1992) "
                "routine DC3D0. Pass z = 0 to evaluate that at the surface.")

    if fault_origin not in FAULT_ORIGINS:
        raise ValueError(
            f"'fault_origin' must be one of {list(FAULT_ORIGINS)}, "
            f"got {fault_origin!r}. (It is ignored for a point source, but is "
            f"still checked so that a typo cannot pass unnoticed.)")

    lo, hi = NU_RANGE
    if not (lo < float(nu) < hi):
        raise ValueError(
            f"Poisson's ratio must satisfy {lo} < nu < {hi}, got {nu!r}. "
            f"This is the range in which the elastic energy is positive "
            f"definite; nu = {hi} is the incompressible limit, where the "
            f"Okada (1985) medium constant 1 - 2*nu vanishes.")

    return coords, params, len(given) == 2




class OkadaWrapper:
    """
    Convenient wrapper class to use functions 
    `SPOINT`, `SRECTF`, `DC3D0` and `DC3D`.

    The class holds no state; it exists to give the three methods a common home.
    """

    def compute(self, coords:dict, params:dict, 
                compute_strain:bool=True, is_degree:bool=True, fault_origin:str="topleft", nu:float=0.25,
                return_iret:bool=False):
        """
        Perform forward computations; given the source parameters, 
        the displacements and/or their spatial derivatives 
        at the station are calculated.

        Currently, multiple station coordinates can be specified, 
        but only one set of source parameters can be specified. 
        If you have multiple sources, 
        you need to call this method multiple times.

        Parameters
        ----------
        coords : dict of torch.Tensor
            `"x"` and `"y"` are required keys, 
            and `"z"` is optional; giving it selects the Okada (1992)
            formulation. Unrecognised keys are rejected.
            Each value must be torch.Tensor of the same shape (`dim` is arbitrary).

        params : dict of torch.Tensor
            `"x_fault"`, `"y_fault"`, `"depth"`, `"strike"`, `"dip"`, `"rake"`
            and `"slip"` are required keys, and `"length"` and `"width"` 
            are optional and
            select a rectangular fault (both) or a point source (neither).
            `"opening"` (tensile / dike opening, in the same units as `"slip"`)
            and `"inflation"` (isotropic point source; needs `"z"` and no
            `"length"`/`"width"`) are optional too and default to zero.
            Unrecognised keys are rejected.
            Each value must be torch.Tensor with dim=0 (scalar tensor).

        compute_strain : bool, default True
            Option to calculate the spatial derivative of the displacement.

        is_degree : bool, default True
            Flag if `"strike"`, `"dip"` and `"rake"`
            are in degree or not (= in radian). 
        
        fault_origin : str, default "topleft"
            In the case of a rectangular fault,
            this flag specifies which point the fault location parameter refers to 
            (ignored for a point source).
            If `fault_origin` is "topleft", then `"x_fault"`, `"y_fault"` and `"depth"` in `params` 
            represent the coordinates of the top left corner of the rectangle.
            If `fault_origin` is "center", then `"x_fault"`, `"y_fault"` and `"depth"` in `params` 
            represent the coordinates of the rectangle's center.
            Other strings cannot be specified.            

        nu : float, default 0.25
            Poisson's ratio.

        return_iret : bool, default False
            Also return the per-station return code.
            `IRET=0` means normal, `IRET=1` means singular (the station
            coincides with the source, or lies on an edge of the fault),
            `IRET=2` means a positive `z` was given.
            Stations with `IRET != 0` are returned as exactly zero, following
            the original FORTRAN; without this flag there is no way to tell
            those apart from a station whose displacement is genuinely zero.


        Returns
        -------
        list of torch.Tensor
            If `compute_strain` is `True`, return is a list of 12 tensors 
            (3 displacements and 9 spatial derivatives):
            [ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz]
            If `False`, return is a list of 3 tensors (displacements only):
            [ux, uy, uz]
            The shape of each tensor is same as that of `coords["x"]` etc.

        iret : torch.Tensor (int), returned only if `return_iret` is `True`
            Return code, with the same shape as `coords["x"]`.
        """

        coords, params, is_rectangular = _validate(coords, params, fault_origin, nu)

        x, y = coords["x"], coords["y"]
        x_fault, y_fault, depth = params["x_fault"], params["y_fault"], params["depth"]
        strike, dip, rake = params["strike"], params["dip"], params["rake"]
        slip = params["slip"]
        opening = params.get("opening", 0.0)      # tensile / dike opening
        inflation = params.get("inflation", 0.0)  # isotropic point source


        # ---- 1. setup ----
        ss, cs, sd, cd, u_strike, u_dip = setup(strike, dip, rake, slip, is_degree)
        xx =  (x - x_fault) * ss + (y - y_fault) * cs
        yy = -(x - x_fault) * cs + (y - y_fault) * ss 


        alpha_1985 = 1 - 2.0 * nu         # MYU/(LAMBDA+MYU), equal to 1/2 if Poisson medium
        alpha_1992 = 1 / (2.0 * (1 - nu)) # (LAMBDA+MYU)/(LAMBDA+2*MYU), equal to 2/3 if Poisson medium

        # ---- 2. model switch ----
        if is_rectangular:
            # rectangular fault 
            length, width = params["length"], params["width"]

            if "z" in coords:
                # DC3D
                z = coords["z"]
                if fault_origin == "topleft":
                    out, iret = DC3D(
                        alpha_1992, xx, yy, z, depth, dip, 0.0, length, -width, 0.0, 
                        u_strike, u_dip, opening, compute_strain, is_degree
                    )
                else:   # "center"
                    out, iret = DC3D(
                        alpha_1992, xx, yy, z, depth, dip, -length/2, +length/2, -width/2, +width/2, 
                        u_strike, u_dip, opening, compute_strain, is_degree
                    )
            else:
                # SRECTF
                if fault_origin == "topleft":
                    yy = yy + width * cd
                    dep = depth + width * sd
                    out, iret = SRECTF(
                        alpha_1985, xx, yy, dep, length, width, sd, cd, 
                        u_strike, u_dip, opening, compute_strain, return_iret=True
                    )
                else:   # "center"
                    xx = xx + length / 2
                    yy = yy + width * cd / 2
                    dep = depth + width * sd / 2
                    out, iret = SRECTF(
                        alpha_1985, xx, yy, dep, length, width, sd, cd, 
                        u_strike, u_dip, opening, compute_strain, return_iret=True
                    )

        else:
            # point source
            if "z" in coords:
                # DC3D0
                z = coords["z"]
                out, iret = DC3D0(
                    alpha_1992, xx, yy, z, depth, dip, 
                    u_strike, u_dip, opening, inflation, compute_strain, is_degree
                )
            else:
                # SPOINT
                out, iret = SPOINT(
                    alpha_1985, xx, yy, depth, sd, cd, 
                    u_strike, u_dip, opening, compute_strain, return_iret=True
                )      

    
        # ---- 3. inversely rotate coordinate ----
        if compute_strain:
            if "z" in coords:
                ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz = out
            else:
                ux, uy, uz, uxx, uxy, uyx, uyy, uzx, uzy = out
                # derived from surface boundary condition (Okada 1985; eq.42)
                uxz = -uzx
                uyz = -uzy
                uzz = -(uxx + uyy) * nu / (1 - nu) 
            ux, uy, uz = rotate_vector(
                ux, uy, uz, ss, cs
            )
            uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz = rotate_tensor(
                uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz, ss, cs
            )
            result = [ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz]
        else:
            ux, uy, uz = out 
            ux, uy, uz = rotate_vector(
                ux, uy, uz, ss, cs
            )
            result = [ux, uy, uz]

        # Rotating a zero vector/tensor leaves it zero, so the stations flagged
        # by `iret` are still exactly zero here.
        return (result, iret) if return_iret else result
        


    def gradient(self, coords:dict, params:dict, arg:str, 
                 compute_strain:bool=True, is_degree:bool=True, fault_origin:str="topleft", nu:float=0.25):
        """
        Calculate gradient with respect to specified `arg` 
        (one of coordinates or parameters) at the station, 
        given the source parameters.

        Currently, only a single `arg` can be specified.
        If you want to get gradient with respect to multiple args, 
        you need to call this method multiple times.

        
        Parameters
        ----------
        coords : dict of torch.Tensor
            `"x"` and `"y"` are required keys, 
            and `"z"` is optional; giving it selects the Okada (1992)
            formulation. Unrecognised keys are rejected.
            Each value must be torch.Tensor of the same shape (`dim` is arbitrary).

        params : dict of torch.Tensor
            `"x_fault"`, `"y_fault"`, `"depth"`, `"strike"`, `"dip"`, `"rake"` 
            and `"slip"` are required keys, and `"length"` and `"width"` 
            are optional and
            select a rectangular fault (both) or a point source (neither).
            `"opening"` (tensile / dike opening, in the same units as `"slip"`)
            and `"inflation"` (isotropic point source; needs `"z"` and no
            `"length"`/`"width"`) are optional too and default to zero.
            Unrecognised keys are rejected.
            Each value must be torch.Tensor with dim=0 (scalar tensor).

        arg : str
            Name of the variable to be differentiated. 
            This should be a key of `coords` or `params`.

        compute_strain : bool, default True
            Option to calculate the spatial derivative of the displacement.

        is_degree : bool, default True
            Flag if `"strike"`, `"dip"` and `"rake"` 
            are in degree or not (= in radian). 

        fault_origin : str, default "topleft"
            In the case of a rectangular fault,
            this flag specifies which point the fault location parameter refers to 
            (ignored for a point source).
            If `fault_origin` is "topleft", then `"x_fault"`, `"y_fault"` and `"depth"` in `params` 
            represent the coordinates of the top left corner of the rectangle.
            If `fault_origin` is "center", then `"x_fault"`, `"y_fault"` and `"depth"` in `params` 
            represent the coordinates of the rectangle's center.
            Other strings cannot be specified.    

        nu : float, default 0.25
            Poisson's ratio.


        Returns
        -------
        list of torch.Tensor
            Same as the `compute` method, 
            but each tensor is differentiated by `arg`.
        """

        coords, params, _ = _validate(coords, params, fault_origin, nu)

        if ("z" in coords) and (arg in ["x", "y", "z"]):
            x, y, z = coords["x"], coords["y"], coords["z"]
            xx, yy, zz = x.flatten(), y.flatten(), z.flatten()
            argnum = COORDINATE_AXIS[arg]

            def _fn(x, y, z):
                coords2 = {"x": x, "y": y, "z": z}
                return self.compute(coords2, params, compute_strain, is_degree, fault_origin, nu)

            _fn_grad = vmap(jacfwd(_fn, argnums=argnum))
            grads = _fn_grad(xx, yy, zz)

            return [g.reshape(x.shape) for g in grads]

        elif ("z" not in coords) and (arg in ["x", "y"]):
            x, y = coords["x"], coords["y"]
            xx, yy = x.flatten(), y.flatten()
            argnum = COORDINATE_AXIS[arg]

            def _fn(x, y):
                coords2 = {"x": x, "y": y}
                return self.compute(coords2, params, compute_strain, is_degree, fault_origin, nu)
            
            _fn_grad = vmap(jacfwd(_fn, argnums=argnum))
            grads = _fn_grad(xx, yy)

            return [g.reshape(x.shape) for g in grads]
        elif arg in params:
            p = params[arg]

            def _fn(p):
                params2 = params.copy()
                params2[arg] = p
                return self.compute(coords, params2, compute_strain, is_degree, fault_origin, nu)

            p = p.detach()

            return jacfwd(_fn)(p)
        else:
            raise ValueError(
                f"arg={arg!r} is not differentiable here: it must be a key of "
                f"'coords' {sorted(coords)} or of 'params' {sorted(params)}.")
            
            



    def hessian(self, coords:dict, params:dict, arg1:str, arg2:str, 
                compute_strain:bool=True, is_degree:bool=True, fault_origin:str="topleft", nu:float=0.25):
        """
        Calculate hessian (2nd-order derivatives) with respect to 
        specified `arg1` and `arg2` at the station, 
        given the source parameters.

        
        Parameters
        ----------        
        coords : dict of torch.Tensor
            `"x"` and `"y"` are required keys, 
            and `"z"` is optional; giving it selects the Okada (1992)
            formulation. Unrecognised keys are rejected.
            Each value must be torch.Tensor of the same shape (`dim` is arbitrary).

        params : dict of torch.Tensor
            `"x_fault"`, `"y_fault"`, `"depth"`, `"strike"`, `"dip"`, `"rake"` 
            and `"slip"` are required keys, and `"length"` and `"width"` 
            are optional and
            select a rectangular fault (both) or a point source (neither).
            `"opening"` (tensile / dike opening, in the same units as `"slip"`)
            and `"inflation"` (isotropic point source; needs `"z"` and no
            `"length"`/`"width"`) are optional too and default to zero.
            Unrecognised keys are rejected.
            Each value must be torch.Tensor with dim=0 (scalar tensor).

        arg1, arg2 : str
            Names of the variable to be differentiated. 
            These should be keys of `coords` or `params`.
            Both `arg1` and `arg2` must be variables of the same kind; 
            both must be `coords` or both must be `params`.

        compute_strain : bool, default True
            Option to calculate the spatial derivative of the displacement.

        is_degree : bool, default True
            Flag if `"strike"`, `"dip"` and `"rake"`
            are in degree or not (= in radian). 

        fault_origin : str, default "topleft"
            In the case of a rectangular fault,
            this flag specifies which point the fault location parameter refers to 
            (ignored for a point source).
            If `fault_origin` is "topleft", then `"x_fault"`, `"y_fault"` and `"depth"` in `params` 
            represent the coordinates of the top left corner of the rectangle.
            If `fault_origin` is "center", then `"x_fault"`, `"y_fault"` and `"depth"` in `params` 
            represent the coordinates of the rectangle's center.
            Other strings cannot be specified.    

        nu : float, default 0.25
            Poisson's ratio. 


        Returns
        -------
        list of torch.Tensor
            Same as the `compute` method, 
            but each tensor is differentiated by `arg1` and `arg2`.
        """


        coords, params, _ = _validate(coords, params, fault_origin, nu)

        # Report the *actual* problem: an arg that is simply absent used to be
        # rejected with the "both must be of the same kind" message, which sent
        # the reader looking for the wrong mistake.
        for label, arg in (("arg1", arg1), ("arg2", arg2)):
            if (arg not in coords) and (arg not in params):
                raise ValueError(
                    f"{label}={arg!r} is not a key of 'coords' {sorted(coords)} "
                    f"or of 'params' {sorted(params)}.")
        if not ((arg1 in coords and arg2 in coords)
                or (arg1 in params and arg2 in params)):
            raise ValueError(
                f"arg1={arg1!r} and arg2={arg2!r} must both be coordinates or "
                f"both be source parameters; mixed second derivatives "
                f"(coordinate x parameter) are not supported.")


        if ("z" in coords) and (arg1 in ["x", "y", "z"]) and (arg2 in ["x", "y", "z"]):
            x, y, z = coords["x"], coords["y"], coords["z"]
            xx, yy, zz = x.flatten(), y.flatten(), z.flatten()

            argnum1, argnum2 = COORDINATE_AXIS[arg1], COORDINATE_AXIS[arg2]

            def _fn(x, y, z):
                coords2 = {"x": x, "y": y, "z": z}
                return self.compute(coords2, params, compute_strain, is_degree, fault_origin, nu)
            
            _fn_hessian = vmap(jacfwd(jacfwd(_fn, argnums=argnum2), argnums=argnum1))
            hessians = _fn_hessian(xx, yy, zz)

            return [h.reshape(x.shape) for h in hessians]

        elif ("z" not in coords) and (arg1 in ["x", "y"]) and (arg2 in ["x", "y"]):
            x, y = coords["x"], coords["y"]
            xx, yy = x.flatten(), y.flatten()

            argnum1, argnum2 = COORDINATE_AXIS[arg1], COORDINATE_AXIS[arg2]

            def _fn(x, y):
                coords2 = {"x": x, "y": y}
                return self.compute(coords2, params, compute_strain, is_degree, fault_origin, nu)
            
            _fn_hessian = vmap(jacfwd(jacfwd(_fn, argnums=argnum2), argnums=argnum1))
            hessians = _fn_hessian(xx, yy)

            return [h.reshape(x.shape) for h in hessians]

        elif (arg1 in params) and (arg2 in params):

            if arg1 == arg2:
                p = params[arg1]

                def _fn(p):
                    params2 = params.copy()
                    params2[arg1] = p
                    return self.compute(coords, params2, compute_strain, is_degree, fault_origin, nu)

                p = p.detach()
                return jacfwd(jacfwd(_fn))(p)
            else:
                p1 = params[arg1]
                p2 = params[arg2]

                def _fn(p1, p2):
                    params2 = params.copy()
                    params2[arg1] = p1
                    params2[arg2] = p2
                    return self.compute(coords, params2, compute_strain, is_degree, fault_origin, nu)

                p1 = p1.detach()
                p2 = p2.detach()

                return jacfwd(jacfwd(_fn, argnums=1), argnums=0)(p1, p2)
        else:
            raise ValueError(
                f"the combination arg1={arg1!r}, arg2={arg2!r} is not supported.")
        