import torch

def setup(strike, dip, rake, slip):

    ss = torch.sin(torch.deg2rad(strike))
    cs = torch.cos(torch.deg2rad(strike))
    
    sd = torch.sin(torch.deg2rad(dip))
    cd = torch.cos(torch.deg2rad(dip))

    # if dip \approx ±90 degree then set sin(dip)=sign(sd) and cos(dip)=0.
    mask = torch.abs(cd) < 1e-3
    sd = torch.where(mask, torch.sign(sd), sd)
    cd = torch.where(mask, torch.tensor(0.0), cd)

    u_strike  = slip * torch.cos(torch.deg2rad(rake))
    u_dip     = slip * torch.sin(torch.deg2rad(rake))
    u_tensile = 0.0

    return [ss, cs, sd, cd, u_strike, u_dip, u_tensile]



def rotate_vector(ux, uy, uz, s, c):
    """
    rotate displacement vector 
    from old coordinate (x-axis is parallel to strike)
    to new coordinate (x & y correspond to east & north)
    """

    Ux = ux * s - uy * c
    Uy = ux * c + uy * s
    Uz = uz

    return [Ux, Uy, Uz]



def rotate_tensor(uxx, uxy, uxz, uyx, uyy, uyz, uzx, uzy, uzz, s, c):
    """
    rotate displacement gradient tensor
    from old coordinate (x-axis is parallel to strike)
    to new coordinate (x & y correspond to east & north)
    """

    Uxx = s * (s * uxx - c * uxy) - c * (s * uyx - c * uyy)
    Uxy = s * (c * uxx + s * uxy) - c * (c * uyx + s * uyy)
    Uxz = s * uxz                 - c * uyz
    Uyx = c * (s * uxx - c * uxy) + s * (s * uyx - c * uyy)
    Uyy = c * (c * uxx + s * uxy) + s * (c * uyx + s * uyy)
    Uyz = c * uxz                 + s * uyz
    Uzx = s * uzx - c * uzy
    Uzy = c * uzx + s * uzy 
    Uzz = uzz
    
    return [Uxx, Uxy, Uxz, Uyx, Uyy, Uyz, Uzx, Uzy, Uzz]
