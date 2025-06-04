import torch
from torch.func import jacfwd, vmap
from .okada1985 import SPOINT, SRECTF
from .okada1992 import DC3D0, DC3D
from .geometry import setup, rotate_vector, rotate_tensor


alpha_1985 = 0.5 # MYU/(LAMBDA+MYU)
alpha_1992 = 2.0/3.0 # (LAMBDA+MYU)/(LAMBDA+2*MYU)


class OkadaWrapper:
    def __init__(self):
        pass

    def compute(self, coords:dict, params:dict, compute_strain:bool):

        assert ("x" in coords) and ("y" in coords), f"'coords' require x and y."
        assert ("x_ref" in params) and ("y_ref" in params) and ("depth" in params) and \
            ("strike" in params) and ("dip" in params) and ("rake" in params) and ("slip" in params), \
            "'params' require x_ref, y_ref, depth, strike, dip, rake and slip."

        x, y = coords["x"], coords["y"]
        assert x.shape == y.shape, "shepe of x and y must be same."
        x_ref, y_ref, depth = params["x_ref"], params["y_ref"], params["depth"]
        strike, dip, rake = params["strike"], params["dip"], params["rake"]
        slip = params["slip"]


        # ---- 1. setup ----
        ss, cs, sd, cd, u_strike, u_dip = setup(strike, dip, rake, slip)
        xx =  (x - x_ref) * ss + (y - y_ref) * cs
        yy = -(x - x_ref) * cs + (y - y_ref) * ss 


        # ---- 2. model switch ----
        if ("length" in params) and ("width" in params):
            # recangular fault 
            
            length, width = params["length"], params["width"]

            if "z" in coords:
                z = coords["z"]
                assert x.shape == y.shape == z.shape, "shepe of x, y and z must be same."
                out = DC3D(
                    alpha_1992, xx, yy, z, dep, 0.0, length, -width, 0.0, u_strike, u_dip, 0.0, compute_strain
                )
            else:
                yy = yy + width * cd
                dep = depth + width * sd
                out = SRECTF(
                    alpha_1985, xx, yy, dep, length, width, sd, cd, u_strike, u_dip, 0.0, compute_strain
                )
        else:
            # point source
            if "z" in coords:
                z = coords["z"]
                assert x.shape == y.shape == z.shape, "shepe of x, y and z must be same."
                out = DC3D0(
                    alpha_1992, xx, yy, z, depth, dip, u_strike, u_dip, 0.0, 0.0, compute_strain
                )
            else:
                out = SPOINT(
                    alpha_1985, xx, yy, depth, sd, cd, u_strike, u_dip, 0.0, compute_strain
                )      

    
        # ---- 3. inversely rotate coordinate ----
        if compute_strain:
            ux, uy, uz, uxx, uxy, uyx, uyy, uzx, uzy = out

            # derived from surface boundary condition (Okada 1985; eq.42)
            uxz = - uzx
            uyz = - uzy
            uzz = - (uxx + uyy) / 3.0 # assume Poisson medium

            ux, uy, uz = rotate_vector(
                ux, uy, uz, ss, cs
            )
            uxx, uxy, uxz, uyx, uyy, uyz, uzx, uzy, uzz = rotate_tensor(
                uxx, uxy, uxz, uyx, uyy, uyz, uzx, uzy, uzz, ss, cs
            )
            return [ux, uy, uz, uxx, uxy, uxz, uyx, uyy, uyz, uzx, uzy, uzz]
        
        else:
            ux, uy, uz = out 
            ux, uy, uz = rotate_vector(
                ux, uy, uz, ss, cs
            )
            return [ux, uy, uz]
        


    def gradient(self, coords:dict, params:dict, compute_strain:bool, arg:str):

        assert ("x" in coords) and ("y" in coords), f"'coords' require 'x' and 'y'."
        assert ("x_ref" in params) and ("y_ref" in params) and ("depth" in params) and \
            ("strike" in params) and ("dip" in params) and ("rake" in params) and ("slip" in params), \
            "'params' require 'x_ref', 'y_ref', 'depth', 'strike', 'dip', 'rake' and 'slip'."



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
                return self.compute(coords2, params, compute_strain)

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
                return self.compute(coords2, params, compute_strain)
            
            _fn_grad = vmap(jacfwd(_fn, argnums=argnum))
            grads = _fn_grad(xx, yy)

            return [g.reshape(x.shape) for g in grads]

        elif arg in params:
            p = params[arg]

            def _fn(p):
                params2 = params.copy()
                params2[arg] = p
                return self.compute(coords, params2, compute_strain)

            p = p.detach()

            return jacfwd(_fn)(p)

        else:
            raise ValueError(f"Invalid arg is specified: '{arg}'.")
            
            



    def hessian(self, coords: dict, params: dict, compute_strain: bool, arg1: str, arg2: str):

        assert ("x" in coords) and ("y" in coords), "'coords' require x and y."
        assert ("x_ref" in params) and ("y_ref" in params) and ("depth" in params) and \
            ("strike" in params) and ("dip" in params) and ("rake" in params) and ("slip" in params), \
            "'params' require x_ref, y_ref, depth, strike, dip, rake and slip."
        
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
                return self.compute(coords2, params, compute_strain)
            
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
                return self.compute(coords2, params, compute_strain)
            
            _fn_hessian = vmap(jacfwd(jacfwd(_fn, argnums=argnum2), argnums=argnum1))
            hessians = _fn_hessian(xx, yy)

            return [h.reshape(x.shape) for h in hessians]

        elif (arg1 in params) and (arg2 in params):

            if arg1 == arg2:
                p = params[arg1]

                def _fn(p):
                    params2 = params.copy()
                    params2[arg1] = p
                    return self.compute(coords, params2, compute_strain)

                p = p.detach()
                return jacfwd(jacfwd(_fn))(p)

            else:

                p1 = params[arg1]
                p2 = params[arg2]

                def _fn(p1, p2):
                    params2 = params.copy()
                    params2[arg1] = p1
                    params2[arg2] = p2
                    return self.compute(coords, params2, compute_strain)

                p1 = p1.detach()
                p2 = p2.detach()

                return jacfwd(jacfwd(_fn, argnums=1), argnums=0)(p1, p2)

            
        else:
            raise ValueError(f"combination of arg '{arg1}' and '{arg2}' is not supported.")
        