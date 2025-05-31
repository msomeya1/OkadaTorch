import torch
from okada1985 import SPOINT, SRECTF
# from okada1992 import DC3D0, DC3D
from utils import setup, rotate_vector, rotate_tensor


class OkadaWrapper:
    def __init__(self):
        pass

    def compute(self, coords:dict, params:dict, compute_derivative:bool):

        if "z" in coords:
            raise NotImplementedError("okada1992 is not yet implemented")

        x = coords["x"]
        y = coords["y"]
        x_ref = params["x_ref"]
        y_ref = params["y_ref"]
        depth = params["depth"]
        strike = params["strike"]
        dip = params["dip"]
        rake = params["rake"]
        slip = params["slip"]


        # ---- 1. setup ----
        ss, cs, sd, cd, u_strike, u_dip, u_tensile = setup(strike, dip, rake, slip)


        # ---- 2. model switch ----
        if ("length" in params) and ("width" in params):
            length = params["length"]
            width = params["width"]

            # rotate coordinate
            xx =  (x - x_ref) * ss + (y - y_ref) * cs
            yy = -(x - x_ref) * cs + (y - y_ref) * ss + width * cd
            dep = depth + width * sd

            out = SRECTF(
                0.5, xx, yy, dep, length, width, sd, cd, u_strike, u_dip, u_tensile, compute_derivative
            )
        else:
            # rotate coordinate
            xx =  (x - x_ref) * ss + (y - y_ref) * cs
            yy = -(x - x_ref) * cs + (y - y_ref) * ss 

            out = SPOINT.compute(
                0.5, xx, yy, depth, sd, cd, u_strike, u_dip, u_tensile, compute_derivative
            )      

    
        # ---- 3. inversely rotate coordinate ----
        if compute_derivative:
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