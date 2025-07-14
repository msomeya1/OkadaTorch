import torch
from torch.func import jacfwd, vmap
from .okada1985 import SPOINT, SRECTF
from .okada1992 import DC3D0, DC3D
from .geometry import setup, rotate_vector, rotate_tensor




class OkadaWrapper:
    """
    Convenient wrapper class to use functions 
    `SPOINT`, `SRECTF`, `DC3D0` and `DC3D`.
    """
    def __init__(self):
        pass

    def compute(self, coords:dict, params:dict, 
                compute_strain:bool=True, is_degree:bool=True, fault_origin:str="topleft", nu:float=0.25):
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
            and `"z"` is optional (all other keys are ignored).
            Each value must be torch.Tensor of the same shape (`dim` is arbitrary).

        params : dict of torch.Tensor
            `"x_fault"`, `"y_fault"`, `"depth"`, `"strike"`, `"dip"`, `"rake"`
            and `"slip"` are required keys, and `"length"` and `"width"` 
            are optional (all other keys are ignored).
            Each value must be torch.Tensor with dim=0 (scaler tensor).

        compute_strain : bool, dafault True
            Option to calculate the spatial derivative of the displacement.

        is_degree : bool, dafault True
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
            If `compute_strain` is `True`, return is a list of 12 tensors 
            (3 displacements and 9 spatial derivatives):
            [ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz]
            If `False`, return is a list of 3 tensors (displacements only):
            [ux, uy, uz]
            The shape of each tensor is same as that of `coords["x"]` etc.
        """

        assert ("x" in coords) and ("y" in coords), "'coords' requires 'x' and 'y'."
        assert ("x_fault" in params) and ("y_fault" in params) and ("depth" in params) and \
            ("strike" in params) and ("dip" in params) and ("rake" in params) and ("slip" in params), \
            "'params' requires 'x_fault', 'y_fault', 'depth', 'strike', 'dip', 'rake' and 'slip'."

        x, y = coords["x"], coords["y"]
        assert x.shape == y.shape, "shepe of x and y must be same."
        x_fault, y_fault, depth = params["x_fault"], params["y_fault"], params["depth"]
        strike, dip, rake = params["strike"], params["dip"], params["rake"]
        slip = params["slip"]


        # ---- 1. setup ----
        ss, cs, sd, cd, u_strike, u_dip = setup(strike, dip, rake, slip, is_degree)
        xx =  (x - x_fault) * ss + (y - y_fault) * cs
        yy = -(x - x_fault) * cs + (y - y_fault) * ss 


        alpha_1985 = 1 - 2.0 * nu         # MYU/(LAMBDA+MYU), equal to 1/2 if Poisson medium
        alpha_1992 = 1 / (2.0 * (1 - nu)) # (LAMBDA+MYU)/(LAMBDA+2*MYU), equal to 2/3 if Poisson medium

        # ---- 2. model switch ----
        if ("length" in params) and ("width" in params):
            # recangular fault 
            length, width = params["length"], params["width"]

            if "z" in coords:
                # DC3D
                z = coords["z"]
                assert x.shape == y.shape == z.shape, "shepe of x, y and z must be same."
                if fault_origin == "topleft":
                    out, _ = DC3D(
                        alpha_1992, xx, yy, z, depth, dip, 0.0, length, -width, 0.0, 
                        u_strike, u_dip, 0.0, compute_strain, is_degree
                    )
                elif fault_origin == "center":
                    out, _ = DC3D(
                        alpha_1992, xx, yy, z, depth, dip, -length/2, +length/2, -width/2, +width/2, 
                        u_strike, u_dip, 0.0, compute_strain, is_degree
                    )
                else:
                    raise ValueError("'fault_origin' must be either 'topleft' or 'center'.")
            else:
                # SRECTF
                if fault_origin == "topleft":
                    yy = yy + width * cd
                    dep = depth + width * sd
                    out = SRECTF(
                        alpha_1985, xx, yy, dep, length, width, sd, cd, 
                        u_strike, u_dip, 0.0, compute_strain
                    )
                elif fault_origin == "center":
                    xx = xx + length / 2
                    yy = yy + width * cd / 2
                    dep = depth + width * sd / 2
                    out = SRECTF(
                        alpha_1985, xx, yy, dep, length, width, sd, cd, 
                        u_strike, u_dip, 0.0, compute_strain
                    )

        else:
            # point source
            if "z" in coords:
                # DC3D0
                z = coords["z"]
                assert x.shape == y.shape == z.shape, "shepe of x, y and z must be same."
                out, _ = DC3D0(
                    alpha_1992, xx, yy, z, depth, dip, 
                    u_strike, u_dip, 0.0, 0.0, compute_strain, is_degree
                )
            else:
                # SPOINT
                out = SPOINT(
                    alpha_1985, xx, yy, depth, sd, cd, 
                    u_strike, u_dip, 0.0, compute_strain
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
            return [ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz]
        else:
            ux, uy, uz = out 
            ux, uy, uz = rotate_vector(
                ux, uy, uz, ss, cs
            )
            return [ux, uy, uz]
        


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
            and `"z"` is optional (all other keys are ignored).
            Each value must be torch.Tensor of the same shape (`dim` is arbitrary).

        params : dict of torch.Tensor
            `"x_fault"`, `"y_fault"`, `"depth"`, `"strike"`, `"dip"`, `"rake"` 
            and `"slip"` are required keys, and `"length"` and `"width"` 
            are optional (all other keys are ignored).
            Each value must be torch.Tensor with dim=0 (scaler tensor).

        arg : str
            Name of the variable to be differentiated. 
            This should be a key of `coords` or `params`.

        compute_strain : bool, dafault True
            Option to calculate the spatial derivative of the displacement.

        is_degree : bool, dafault True
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

        assert ("x" in coords) and ("y" in coords), f"'coords' requires 'x' and 'y'."
        assert ("x_fault" in params) and ("y_fault" in params) and ("depth" in params) and \
            ("strike" in params) and ("dip" in params) and ("rake" in params) and ("slip" in params), \
            "'params' requires 'x_fault', 'y_fault', 'depth', 'strike', 'dip', 'rake' and 'slip'."


        if ("z" in coords) and (arg in ["x", "y", "z"]):
            x, y, z = coords["x"], coords["y"], coords["z"]
            assert x.shape == y.shape == z.shape, "shepe of x, y and z must be same."
            xx, yy, zz = x.flatten(), y.flatten(), z.flatten()
            if arg=="x":
                argnum = 0 
            elif arg=="y":
                argnum = 1
            else:
                argnum = 2

            def _fn(x, y, z):
                coords2 = {"x": x, "y": y, "z": z}
                return self.compute(coords2, params, compute_strain, is_degree, fault_origin, nu)

            _fn_grad = vmap(jacfwd(_fn, argnums=argnum))
            grads = _fn_grad(xx, yy, zz)

            return [g.reshape(x.shape) for g in grads]

        elif ("z" not in coords) and (arg in ["x", "y"]):
            x, y = coords["x"], coords["y"]
            assert x.shape == y.shape, "shepe of x and y must be same."
            xx, yy = x.flatten(), y.flatten()
            if arg=="x":
                argnum = 0 
            elif arg=="y":
                argnum = 1

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
            raise ValueError(f"Invalid arg is specified: '{arg}'.")
            
            



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
            and `"z"` is optional (all other keys are ignored).
            Each value must be torch.Tensor of the same shape (`dim` is arbitrary).

        params : dict of torch.Tensor
            `"x_fault"`, `"y_fault"`, `"depth"`, `"strike"`, `"dip"`, `"rake"` 
            and `"slip"` are required keys, and `"length"` and `"width"` 
            are optional (all other keys are ignored).
            Each value must be torch.Tensor with dim=0 (scaler tensor).

        arg1, arg2 : str
            Names of the variable to be differentiated. 
            These should be keys of `coords` or `params`.
            Both `arg1` and `arg2` must be variables of the same kind; 
            both must be `coords` or both must be `params`.

        compute_strain : bool, dafault True
            Option to calculate the spatial derivative of the displacement.

        is_degree : bool, dafault True
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


        assert ("x" in coords) and ("y" in coords), "'coords' requires 'x' and 'y'."
        assert ("x_fault" in params) and ("y_fault" in params) and ("depth" in params) and \
            ("strike" in params) and ("dip" in params) and ("rake" in params) and ("slip" in params), \
            "'params' requires 'x_fault', 'y_fault', 'depth', 'strike', 'dip', 'rake' and 'slip'."
        
        assert (arg1 in coords and arg2 in coords) or (arg1 in params and arg2 in params), \
            "Both arg1 and arg2 must be variables of the same kind; both must be coords or both must be params."


        if ("z" in coords) and (arg1 in ["x", "y", "z"]) and (arg2 in ["x", "y", "z"]):
            x, y, z = coords["x"], coords["y"], coords["z"]
            assert x.shape == y.shape == z.shape, "shepe of x, y and z must be same."
            xx, yy, zz = x.flatten(), y.flatten(), z.flatten()

            if arg1=="x":
                argnum1 = 0 
            elif arg1=="y":
                argnum1 = 1
            else:
                argnum1 = 2

            if arg2=="x":
                argnum2 = 0 
            elif arg2=="y":
                argnum2 = 1
            else:
                argnum2 = 2

            def _fn(x, y, z):
                coords2 = {"x": x, "y": y, "z": z}
                return self.compute(coords2, params, compute_strain, is_degree, fault_origin, nu)
            
            _fn_hessian = vmap(jacfwd(jacfwd(_fn, argnums=argnum2), argnums=argnum1))
            hessians = _fn_hessian(xx, yy, zz)

            return [h.reshape(x.shape) for h in hessians]

        elif ("z" not in coords) and (arg1 in ["x", "y"]) and (arg2 in ["x", "y"]):
            x, y = coords["x"], coords["y"]
            assert x.shape == y.shape, "shepe of x and y must be same."
            xx, yy = x.flatten(), y.flatten()

            if arg1=="x":
                argnum1 = 0 
            elif arg1=="y":
                argnum1 = 1

            if arg2=="x":
                argnum2 = 0 
            elif arg2=="y":
                argnum2 = 1

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
            raise ValueError(f"combination of arg '{arg1}' and '{arg2}' is not supported.")
        