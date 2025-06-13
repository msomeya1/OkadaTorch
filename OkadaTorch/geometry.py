import torch

EPS = 1.0e-6

def setup(strike, dip, rake, slip, is_degree):
    """
    Calculate sine and cosine of angle variables.
    """

    if is_degree:
        ss = torch.sin(torch.deg2rad(strike))
        cs = torch.cos(torch.deg2rad(strike))
        sd = torch.sin(torch.deg2rad(dip))
        cd = torch.cos(torch.deg2rad(dip))
        u_strike  = slip * torch.cos(torch.deg2rad(rake))
        u_dip     = slip * torch.sin(torch.deg2rad(rake))
    else:
        ss = torch.sin(strike)
        cs = torch.cos(strike)
        sd = torch.sin(dip)
        cd = torch.cos(dip)
        u_strike  = slip * torch.cos(rake)
        u_dip     = slip * torch.sin(rake)


    # if dip≈±90° then set sd=sign(sd) and cd=0.
    if torch.abs(cd) < EPS:
        sd = torch.sign(sd)
        cd = 0.0


    return [ss, cs, sd, cd, u_strike, u_dip]



def rotate_vector(ux, uy, uz, s, c):
    """
    Rotate displacement vector 
    from old coordinate (x-axis is parallel to strike)
    to new coordinate (x & y correspond to east & north).

    Parameters
    ----------
    ux, uy, uz
        Components of displacement vector.
    s, c
        Sine and cosine of strike-angle.

    Returns
    -------
    Ux, Uy, Uz
        Components of rotated displacement vector.
    """

    Ux = ux * s - uy * c
    Uy = ux * c + uy * s
    Uz = uz

    return [Ux, Uy, Uz]



def rotate_tensor(uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz, s, c):
    """
    Rotate displacement gradient tensor
    from old coordinate (x-axis is parallel to strike)
    to new coordinate (x & y correspond to east & north).

    Parameters
    ----------
    uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz
        Components of displacement gradient tensor.
    s, c
        Sine and cosine of strike-angle.
    
    Returns
    -------
    Uxx, Uyx, Uzx, Uxy, Uyy, Uzy, Uxz, Uyz, Uzz
        Components of rotated displacement gradient tensor.
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
    
    return [Uxx, Uyx, Uzx, Uxy, Uyy, Uzy, Uxz, Uyz, Uzz]
