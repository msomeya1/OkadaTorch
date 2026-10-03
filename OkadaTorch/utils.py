from typing import NamedTuple

import torch
from torch import Tensor

PI2 = 2.0 * torch.pi

# --- thresholds -------------------------------------------------------------
# Length comparisons are relative, not absolute.  A fixed `|xi| < 1e-6` means
# "within a millimetre" in kilometres and "within a micron" in metres, so the
# same fault used to give different answers depending on the unit chosen (and
# `R * (R + xi)` was compared against a length, though it is an area).  Scaling
# with the machine epsilon of the working dtype also keeps the guards alive in
# float32, where a float64-sized tolerance never fires.
#
# The factor has to stay small enough to be a noise floor in float32 as well:
# 1e4 works out to 1.2e-3 there, which snaps coordinates over hundreds of metres
# on a 200 km fault and costs three digits of accuracy.
REL_EPS_FACTOR = 1.0e2
EPS_DIP = 1.0e-6         # |cos(dip)| below this selects the vertical-fault formulae
# Below this the inclined-fault formulae keep their value but lose their
# derivative to a cancellation between two 1/cos(dip) terms; `_surrogate_grad`
# supplies it instead.
DIP_GRAD_FLOOR = 1.0e-3


def _rel_eps(reference):
    """Relative tolerance for length comparisons, tied to the working precision.

    Tied to the dtype so that the same code guards float32 and float64: a fixed
    float64-sized tolerance sits below the float32 resolution and never fires.
    """
    return REL_EPS_FACTOR * torch.finfo(reference.dtype).eps


def _length_scale(*values, reference):
    """Largest magnitude among `values`, as a tensor like `reference`.

    Detached: a tolerance must not contribute to the gradient.
    """
    out = None
    for v in values:
        v = torch.abs(torch.as_tensor(v, dtype=reference.dtype,
                                      device=reference.device).detach())
        out = v if out is None else torch.maximum(out, v)
    return out


def _as_tensor_like(value, reference):
    """Return `value` as a tensor sharing `reference`'s dtype and device.

    A no-op for tensors, so autograd history is preserved.  Needed because the
    branchless formulation feeds scalars such as `CD` to `torch.where`, which
    requires its condition to be a tensor.
    """
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _safe_div(numerator, denominator, keep, fill=0.0):
    """`numerator / denominator` where `keep` is true, `fill` elsewhere.

    `torch.where(keep, n / d, fill)` looks equivalent and is not: `torch.where`
    evaluates both arguments, so the discarded branch still divides by zero and
    its `inf` reaches the backward pass as `0 * inf = nan`.  Sanitising the
    denominator first leaves the selected values bit-identical.
    """
    denominator = torch.as_tensor(denominator)
    safe = torch.where(keep, denominator, torch.ones_like(denominator))
    return torch.where(keep, numerator / safe, torch.full_like(safe, fill))


def _safe_log(value, keep, fill=0.0):
    """`log(value)` where `keep` is true, `fill` elsewhere.  See `_safe_div`."""
    value = torch.as_tensor(value)
    safe = torch.where(keep, value, torch.ones_like(value))
    return torch.where(keep, torch.log(safe), torch.full_like(safe, fill))


def _dip_offset(SD, CD, delta=DIP_GRAD_FLOOR):
    """Rotate (SD, CD) away from the vertical by a *constant* angle.

    Returns sin/cos of a dip shifted by a fixed amount, so `d(CD_off)/d(dip) =
    -SD_off` follows from the chain rule; the shift carries no gradient itself.
    The sign is chosen so the rotation always moves away from cos(dip) = 0.
    """
    one = torch.ones_like(CD)
    sign_cd = torch.where(CD >= 0.0, one, -one)
    sign_sd = torch.where(SD >= 0.0, one, -one)
    angle = -sign_cd * sign_sd * delta            # constant w.r.t. autograd
    c, s = torch.cos(angle), torch.sin(angle)
    return SD * c + CD * s, CD * c - SD * s


def _surrogate_grad(use_surrogate, value, surrogate):
    """`value`'s value, `surrogate`'s gradient, wherever `use_surrogate` is true.

    A straight-through estimator, used near a vertical fault where the value and
    the derivative must come from different places: the vertical-fault formula is
    the exact limit but has no cos(dip) dependence to differentiate, while the
    inclined one has the dependence but loses all precision computing it.
    """
    # `value.detach() + (surrogate - surrogate.detach())`, not the more common
    # `surrogate + (value - surrogate).detach()`: the second term is exactly zero
    # here, so the forward value is bit-identical.  The other spelling leaves a
    # rounding error of order ulp(surrogate), and |surrogate| >> |value| is the
    # normal case.
    return torch.where(use_surrogate,
                       value.detach() + (surrogate - surrogate.detach()),
                       value)


def _srectg_A_inclined(ALP, XI, ET, Q, SD, CD, R, X, RD, Y, DLE, xi_nonzero):
    """A1, A3, A4, A5 of `_SRECTG`, inclined-fault form.  `CD` must be non-zero."""
    TD = SD / CD
    A5 = torch.where(
        xi_nonzero,
        ALP * 2.0 / CD * torch.atan(_safe_div(
            ET * (X + Q * CD) + X * (R + X) * SD, XI * (R + X) * CD, xi_nonzero
        )),
        torch.zeros_like(R)
    )
    A4 =  ALP / CD * (torch.log(RD) - SD * DLE)
    A3 =  ALP * ( Y / RD / CD - DLE) + TD * A4
    A1 = -ALP / CD * XI / RD         - TD * A5
    return A1, A3, A4, A5


def _srectg_BC_inclined(ALP, XI, XI2, Q, SD, CD, RD, Y, RRD, RRE):
    """B1, B2, C1, C3 of `_SRECTG`, inclined-fault form.  `CD` must be non-zero."""
    TD = SD / CD
    C1 = ALP / CD * XI * (RRD - SD * RRE)
    C3 = ALP / CD * (Q * RRE - Y * RRD)
    B1 = ALP / CD * (XI2 * RRD - 1.0) / RD - TD * C3
    B2 = ALP / CD * XI * Y * RRD / RD      - TD * C1
    return B1, B2, C1, C3


def _ub_AI_inclined(XI, ET, Q, SD, CD, SDCD, R, X, RD, Y, ALE, xi_nonzero):
    """AI3, AI4 of `_UB`, inclined-fault form.  `CD` must be non-zero."""
    CDCD = CD**2
    AI4 = torch.where(
        xi_nonzero,
        1.0 / CDCD * (XI / RD * SDCD + 2.0 * torch.atan(_safe_div(
            ET * (X + Q * CD) + X * (R + X) * SD, XI * (R + X) * CD, xi_nonzero
        ))),
        torch.zeros_like(R)
    )
    AI3 = (Y * CD / RD - ALE + SD * torch.log(RD)) / CDCD
    return AI3, AI4


def _ub_AJK_inclined(XI, Q, SD, CD, Y, Y11, D11, AJ2, AJ5):
    """AK1, AK3, AJ3, AJ6 of `_UB`, inclined-fault form.  `CD` must be non-zero."""
    AK1 = XI * (D11 - Y11 * SD) / CD
    AK3 = (Q * Y11 - Y * D11) / CD
    AJ3 = (AK1 - AJ2 * SD) / CD
    AJ6 = (AK3 - AJ5 * SD) / CD
    return AK1, AK3, AJ3, AJ6


def _snap(value, scale):
    """Round `value` to exactly zero when it is negligible compared with `scale`.

    `scale` is the length the comparison is relative to, so that a station a
    rounding error away from a fault edge is classified the same as one exactly
    on it whatever unit the caller uses.
    """
    value = torch.as_tensor(value)
    return torch.where(torch.abs(value) < _rel_eps(value) * scale,
                       torch.zeros_like(value), value)


def _on_fault_edge(XI0, XI1, ET0, ET1, Q, scale):
    """Okada's singular-case test: does the station lie on an edge of the fault?

    The condition the original `DC3D` uses to set `IRET = 1`, reused for
    `SPOINT` / `SRECTF`, which have no return code of their own, so that a fault
    reaching the surface behaves the same through either formulation.  `scale`
    is the fault dimensions.
    """
    xi0, xi1, et0, et1, q = (_snap(v, scale) for v in (XI0, XI1, ET0, ET1, Q))
    return torch.logical_and(q == 0.0, torch.logical_or(
        torch.logical_and(xi0 * xi1 <= 0.0, et0 * et1 == 0.0),
        torch.logical_and(et0 * et1 <= 0.0, xi0 * xi1 == 0.0),
    ))


def _dummy_where(flag, *tensors, value=1.0):
    """Replace flagged elements with a harmless constant.

    Zeroing a singular station's output repairs its value but not its gradient:
    `torch.where` hands zero back to the discarded branch, which then evaluates
    `0 * d(nan)/dx`.  Giving those stations a dummy geometry keeps every
    intermediate finite instead.  `xi = eta = q = 1` clears every guard.
    """
    return [torch.where(flag, torch.full_like(t, value), t) for t in tensors]


def _blank_where(flag, tensors):
    """Return `tensors` with the flagged elements set to zero."""
    return [torch.where(flag, torch.zeros_like(t), t) for t in tensors]


def _SRECTG(ALP, XI, ET, Q, SD, CD, DISL1, DISL2, DISL3, compute_strain):
    """
    Indefinite integral of surface displacements, strains and tilts
    due to finite fault in a semi-infinite medium.

    Parameters
    ----------
    ALP : float or torch.Tensor
        Medium constant. myu/(lambda+myu)
    XI, ET, Q : torch.Tensor
        Fault coordinate.
    SD, CD : float or torch.Tensor
        Sin, Cosine of dip-angle. 
        (CD=0.0, SD=+/-1.0 should be given for vertical fault.)
    DISL1, DISL2, DISL3 : float or torch.Tensor
        Strike-, dip- and tensile-dislocation.
    compute_strain : bool
        Option to calculate the spatial derivative of the displacement.
        New in the PyTorch implementation.

    Returns
    -------
    U : list of torch.Tensor
        If `compute_strain` is `True`, 
        U is a list of 3 displacements and 6 spatial derivatives:
        [U1, U2, U3, U11, U12, U21, U22, U31, U32]
        If `False`, U is a list of 3 displacements only:
        [U1, U2, U3]

        U1, U2, U3 : torch.Tensor
            Displacement. unit = (unit of dislocation) 
        U11, U12, U21, U22 : torch.Tensor
            Strain. unit = (unit of dislocation) / (unit of XI,ET,Q)
        U31, U32 : torch.Tensor
            Tilt. unit = (unit of dislocation) / (unit of XI,ET,Q)

    Notes
    -----
    Original FORTRAN code was written by Y.Okada in Jan. 1985.
    PyTorch implementation by M.Someya in 2025.
    """

    # Initialization
    if compute_strain:
        U1, U2, U3, U11, U12, U21, U22, U31, U32 = [torch.zeros_like(XI) for _ in range(9)]
    else:
        U1, U2, U3 = [torch.zeros_like(XI) for _ in range(3)]


    XI2 = XI**2
    ET2 = ET**2
    Q2 = Q**2 
    R2 = XI2 + ET2 + Q2 
    R = torch.sqrt(R2)
    D = ET * SD - Q * CD
    Y = ET * CD + Q * SD
    RET = R + ET
    RET = torch.where(
        RET < 0.0,
        0.0,
        RET
    )
    RD = R + D

    SD = _as_tensor_like(SD, R)
    CD = _as_tensor_like(CD, R)

    q_nonzero = Q != 0.0
    ret_nonzero = RET != 0.0
    TT = torch.atan(_safe_div(XI * ET, Q * R, q_nonzero))   # atan(0) == 0
    RE = _safe_div(1.0, RET, ret_nonzero)
    DLE = torch.where(
        ret_nonzero,
        _safe_log(RET, ret_nonzero),
        -_safe_log(R - ET, ~ret_nonzero)
    )
    # RRX = 1/(R(R+XI)) diverges on the negative extension of the fault edge.
    # The 1992 routines set the identical quantity (X11) to zero there, so the
    # same value is used rather than the unit-dependent constant the first port
    # invented.  The choice is unobservable anyway: R + XI = 0 forces ET = Q = 0
    # (R >= |XI|), hence Y = D = 0, and every use of RRX below carries one of
    # those as a factor.
    rrx_regular = (R + XI) >= _rel_eps(R) * R
    RRX = _safe_div(1.0, R * (R + XI), rrx_regular)

    RRE = RE / R

    # A1..A5 -- the inclined and the vertical formula are both evaluated and then
    # selected, so that no Python-level branch depends on a tensor value.  The
    # divisions by CD are guarded by CD_safe, which is exactly 1 wherever the
    # vertical formula wins, so the selected values are unchanged.
    vertical = CD == 0.0
    CD_safe = torch.where(vertical, torch.ones_like(CD), CD)
    xi_nonzero = XI != 0.0
    X = torch.sqrt(XI2 + Q2)

    # INCLINED FAULT
    TD = SD / CD_safe
    A1_inclined, A3_inclined, A4_inclined, A5_inclined = _srectg_A_inclined(
        ALP, XI, ET, Q, SD, CD_safe, R, X, RD, Y, DLE, xi_nonzero)

    # VERTICAL FAULT
    RD2 = RD**2
    A1_vertical = -ALP / 2.0 * XI * Q / RD2
    A3_vertical =  ALP / 2.0 * (ET / RD + Y * Q / RD2 - DLE)
    A4_vertical = -ALP * Q / RD
    A5_vertical = -ALP * XI * SD / RD

    A1_value = torch.where(vertical, A1_vertical, A1_inclined)
    A3_value = torch.where(vertical, A3_vertical, A3_inclined)
    A4_value = torch.where(vertical, A4_vertical, A4_inclined)
    A5_value = torch.where(vertical, A5_vertical, A5_inclined)

    # Near a vertical fault neither formula can supply the derivative (see
    # `_surrogate_grad`), so it is taken from the inclined formula evaluated at a
    # dip rotated a constant DIP_GRAD_FLOOR away from vertical.  The A-, B- and
    # C-terms need this because they are where 1/CD appears, and therefore where
    # the 1/CD divergences have to cancel.
    ill_conditioned = torch.abs(CD) < DIP_GRAD_FLOOR
    SD_off, CD_off = _dip_offset(SD, CD)
    D_off  = ET * SD_off - Q * CD_off
    Y_off  = ET * CD_off + Q * SD_off
    RD_off = R + D_off
    A1_surr, A3_surr, A4_surr, A5_surr = _srectg_A_inclined(
        ALP, XI, ET, Q, SD_off, CD_off, R, X, RD_off, Y_off, DLE, xi_nonzero)

    A1 = _surrogate_grad(ill_conditioned, A1_value, A1_surr)
    A3 = _surrogate_grad(ill_conditioned, A3_value, A3_surr)
    A4 = _surrogate_grad(ill_conditioned, A4_value, A4_surr)
    A5 = _surrogate_grad(ill_conditioned, A5_value, A5_surr)

    A2 = -ALP * DLE - A3


    if compute_strain:
        RRD = 1.0 / (R * RD)
        AXI = (2.0 * R + XI) * RRX**2 / R
        AET = (2.0 * R + ET) * RRE**2 / R
        R3 = R**3

        # INCLINED FAULT
        B1_inclined, B2_inclined, C1_inclined, C3_inclined = _srectg_BC_inclined(
            ALP, XI, XI2, Q, SD, CD_safe, RD, Y, RRD, RRE)

        # VERTICAL FAULT
        B1_vertical = ALP / 2.0 * Q       / RD2 * (2.0 * XI2 * RRD - 1.0)
        B2_vertical = ALP / 2.0 * XI * SD / RD2 * (2.0 * Q2  * RRD - 1.0)
        C1_vertical = ALP * XI * Q * RRD / RD
        C3_vertical = ALP * SD / RD * (XI2 * RRD - 1.0)

        B1_value = torch.where(vertical, B1_vertical, B1_inclined)
        B2_value = torch.where(vertical, B2_vertical, B2_inclined)
        C1_value = torch.where(vertical, C1_vertical, C1_inclined)
        C3_value = torch.where(vertical, C3_vertical, C3_inclined)

        RRD_off = 1.0 / (R * RD_off)
        B1_surr, B2_surr, C1_surr, C3_surr = _srectg_BC_inclined(
            ALP, XI, XI2, Q, SD_off, CD_off, RD_off, Y_off, RRD_off, RRE)

        B1 = _surrogate_grad(ill_conditioned, B1_value, B1_surr)
        B2 = _surrogate_grad(ill_conditioned, B2_value, B2_surr)
        C1 = _surrogate_grad(ill_conditioned, C1_value, C1_surr)
        C3 = _surrogate_grad(ill_conditioned, C3_value, C3_surr)

        B3 = -ALP * XI * RRE - B2
        B4 = -ALP * (CD / R + Q * SD * RRE) - B1
        C2 = ALP * (-SD / R + Q * CD * RRE) - C3




    # STRIKE-SLIP CONTRIBUTION
    UN = DISL1 / PI2
    REQ = RRE * Q
    U1 = U1 - UN * (REQ * XI + TT          + A1 * SD)
    U2 = U2 - UN * (REQ * Y  + Q * CD * RE + A2 * SD)
    U3 = U3 - UN * (REQ * D  + Q * SD * RE + A4 * SD)

    if compute_strain:
        U11 = U11 + UN * (XI2 * Q * AET - B1 * SD)
        # D / (ET2 + Q2) is 0/0 for a station on the fault-edge line (ET = Q = 0).
        # The original FORTRAN never evaluates it there because the whole
        # strike-slip block sits behind `IF(DISL1.NE.0)`; now that the block is
        # unconditional the division has to be guarded explicitly, otherwise
        # DISL1 = 0 would turn a finite result into 0 * nan.
        et2q2_nonzero = (ET2 + Q2) != 0.0
        U12 = U12 + UN * (XI2 * XI * (_safe_div(D, ET2 + Q2, et2q2_nonzero) / R3
                                      - AET * SD) - B2 * SD)
        U21 = U21 + UN * (XI * Q / R3 * CD + (XI * Q2 * AET - B2) * SD)
        U22 = U22 + UN * (Y * Q / R3 * CD + (Q * SD * (Q2 * AET - 2.0 * RRE) - (XI2 + ET2) / R3 * CD - B4) * SD)
        U31 = U31 + UN * (-XI * Q2 * AET * CD + (XI * Q / R3 - C1) * SD)
        U32 = U32 + UN * (D * Q / R3 * CD + (XI2 * Q * AET * CD - SD / R + Y * Q / R3 - C2) * SD)


    # DIP-SLIP CONTRIBUTION
    UN = DISL2 / PI2
    SDCD = SD * CD
    U1 = U1 - UN * (Q / R                 - A3 * SDCD)
    U2 = U2 - UN * (Y * Q * RRX + CD * TT - A1 * SDCD)
    U3 = U3 - UN * (D * Q * RRX + SD * TT - A5 * SDCD)

    if compute_strain:
        U11 = U11 + UN * (XI * Q / R3                + B3 * SDCD)
        U12 = U12 + UN * (Y * Q / R3  - SD / R       + B1 * SDCD)
        U21 = U21 + UN * (Y * Q / R3  + Q * CD * RRE + B1 * SDCD)
        U22 = U22 + UN * (Y**2  * Q * AXI - (2.0 * Y * RRX + XI * CD * RRE) * SD + B2 * SDCD)
        U31 = U31 + UN * (D * Q / R3  + Q * SD * RRE + C3 * SDCD)
        U32 = U32 + UN * (Y * D * Q * AXI - (2.0 * D * RRX + XI * SD * RRE) * SD + C1 * SDCD)


    # TENSILE-FAULT CONTRIBUTION
    UN = DISL3 / PI2
    SDSD = SD**2
    U1 = U1 + UN * (Q2 * RRE                                - A3 * SDSD)
    U2 = U2 + UN * (-D * Q * RRX - SD * (XI * Q * RRE - TT) - A1 * SDSD)
    U3 = U3 + UN * ( Y * Q * RRX + CD * (XI * Q * RRE - TT) - A5 * SDSD)

    if compute_strain:
        U11 = U11 - UN * (XI * Q2 * AET                    + B3 * SDSD)
        U12 = U12 - UN * (-D * Q / R3 - XI2 * Q * AET * SD + B1 * SDSD)
        U21 = U21 - UN * (Q2 * (CD / R3 + Q * AET * SD)    + B1 * SDSD)
        U22 = U22 - UN * ((Y * CD - D * SD) * Q2 * AXI - 2.0 * Q * SD * CD * RRX - (XI * Q2 * AET - B2) * SDSD)
        U31 = U31 - UN * (Q2 * (SD / R3 - Q * AET * CD) + C3 * SDSD)
        U32 = U32 - UN * ((Y * SD + D * CD) * Q2 * AXI + XI * Q2 * AET * SD * CD - (2.0 * Q * RRX - C1) * SDSD)



    if compute_strain:
        return [U1, U2, U3, U11, U12, U21, U22, U31, U32]
    else:
        return [U1, U2, U3]





def _UA0(X, Y, D, POT1, POT2, POT3, POT4, C0, C1, compute_strain):
    """
    Displacement and strain at depth (Part-A) 
    due to buried point source in a semi-infinite medium.

    Parameters
    ----------
    X, Y, D : torch.Tensor
        Station coordinates in fault system.
    POT1, POT2, POT3, POT4 : float or torch.Tensor
        Strike-, dip-, tensile- and inflate-potency.
    C0, C1
        `MediumConstants` and `PointGeometry` bundles.
    compute_strain : bool
        Option to calculate the spatial derivative of the displacement.
        New in the PyTorch implementation.

    Returns
    -------
    U : list of torch.Tensor
        If `compute_strain` is `True`, 
        U is a list of 3 displacements and 9 spatial derivatives:
        [UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]
        If `False`, U is a list of 3 displacements only:
        [UX, UY, UZ]
    """

    # Initialization
    N_variable = 12 if compute_strain else 3
    U = [torch.zeros_like(X) for _ in range(N_variable)]
    DU = [None] * N_variable


    ALP1, ALP2, SD, CD = C0.ALP1, C0.ALP2, C0.SD, C0.CD
    P, Q, S, T, XY, X2, R3, QR = C1.P, C1.Q, C1.S, C1.T, C1.XY, C1.X2, C1.R3, C1.QR


    if compute_strain:
        S2D, C2D = C0.S2D, C0.C2D
        R5, QRX, A3, A5, B3, C3 = C1.R5, C1.QRX, C1.A3, C1.A5, C1.B3, C1.C3 
        UY, VY, WY, UZ, VZ, WZ = C1.UY, C1.VY, C1.WY, C1.UZ, C1.VZ, C1.WZ   


    # STRIKE-SLIP CONTRIBUTION
    DU[ 0] =  ALP1 * Q / R3      + ALP2 * X2    * QR
    DU[ 1] =  ALP1 * X / R3 * SD + ALP2 * XY    * QR
    DU[ 2] = -ALP1 * X / R3 * CD + ALP2 * X * D * QR

    if compute_strain:
        DU[ 3] = X * QR * (-ALP1 + ALP2 * (1.0 + A5))
        DU[ 4] =  ALP1 * A3 / R3 * SD + ALP2 * Y * QR * A5
        DU[ 5] = -ALP1 * A3 / R3 * CD + ALP2 * D * QR * A5
        DU[ 6] =  ALP1 * (SD / R3 - Y * QR) + ALP2 * 3.0 * X2 / R5 * UY
        DU[ 7] = 3.0 * X / R5 * (-ALP1 * Y * SD + ALP2 * (Y * UY + Q))
        DU[ 8] = 3.0 * X / R5 * ( ALP1 * Y * CD + ALP2 * D * UY)
        DU[ 9] = ALP1 * (CD / R3 + D * QR) + ALP2 * 3.0 * X2 / R5 * UZ
        DU[10] = 3.0 * X / R5 * ( ALP1 * D * SD + ALP2 * Y * UZ)
        DU[11] = 3.0 * X / R5 * (-ALP1 * D * CD + ALP2 * (D * UZ - Q))

    for I in range(N_variable):
        U[I] = U[I] + POT1 / PI2 * DU[I]



    # DIP-SLIP CONTRIBUTION
    DU[ 0] =                  ALP2 * X * P * QR
    DU[ 1] =  ALP1 * S / R3 + ALP2 * Y * P * QR
    DU[ 2] = -ALP1 * T / R3 + ALP2 * D * P * QR

    if compute_strain:
        DU[ 3] =                                         ALP2 * P * QR * A5
        DU[ 4] = -ALP1 * 3.0 * X * S / R5              - ALP2 * Y * P * QRX
        DU[ 5] =  ALP1 * 3.0 * X * T / R5              - ALP2 * D * P * QRX
        DU[ 6] =                                         ALP2 * 3.0 * X / R5 * VY
        DU[ 7] =  ALP1 * (S2D / R3 - 3.0 * Y * S / R5) + ALP2 * (3.0 * Y / R5 * VY + P * QR)
        DU[ 8] = -ALP1 * (C2D / R3 - 3.0 * Y * T / R5) + ALP2 * 3.0 * D / R5 * VY
        DU[ 9] =                                         ALP2 * 3.0 * X / R5 * VZ
        DU[10] =  ALP1 * (C2D / R3 + 3.0 * D * S / R5) + ALP2 * 3.0 * Y / R5 * VZ
        DU[11] =  ALP1 * (S2D / R3 - 3.0 * D * T / R5) + ALP2 * (3.0 * D / R5 * VZ - P * QR)
    
    for I in range(N_variable):
        U[I] = U[I] + POT2 / PI2 * DU[I]



    # TENSILE-FAULT CONTRIBUTION
    DU[ 0] = ALP1 * X / R3 - ALP2 * X * Q * QR
    DU[ 1] = ALP1 * T / R3 - ALP2 * Y * Q * QR
    DU[ 2] = ALP1 * S / R3 - ALP2 * D * Q * QR

    if compute_strain:
        DU[ 3] =  ALP1 * A3 / R3                       - ALP2 * Q * QR * A5
        DU[ 4] = -ALP1 * 3.0 * X * T / R5              + ALP2 * Y * Q * QRX
        DU[ 5] = -ALP1 * 3.0 * X * S / R5              + ALP2 * D * Q * QRX
        DU[ 6] = -ALP1 * 3.0 * XY / R5                 - ALP2 * X * QR * WY
        DU[ 7] =  ALP1 * (C2D / R3 - 3.0 * Y * T / R5) - ALP2 * (Y * WY + Q) * QR
        DU[ 8] =  ALP1 * (S2D / R3 - 3.0 * Y * S / R5) - ALP2 * D * QR * WY
        DU[ 9] =  ALP1 * 3.0 * X * D / R5              - ALP2 * X * QR * WZ
        DU[10] = -ALP1 * (S2D / R3 - 3.0 * D * T / R5) - ALP2 * Y * QR * WZ
        DU[11] =  ALP1 * (C2D / R3 + 3.0 * D * S / R5) - ALP2 * (D * WZ - Q) * QR

    for I in range(N_variable):
        U[I] = U[I] + POT3 / PI2 * DU[I]



    # INFLATE SOURCE CONTRIBUTION
    DU[ 0] = -ALP1 * X / R3
    DU[ 1] = -ALP1 * Y / R3
    DU[ 2] = -ALP1 * D / R3

    if compute_strain:
        DU[ 3] = -ALP1 * A3 / R3
        DU[ 4] =  ALP1 * 3.0 * XY / R5
        DU[ 5] =  ALP1 * 3.0 * X * D / R5
        DU[ 6] =  DU[4]
        DU[ 7] = -ALP1 * B3 / R3
        DU[ 8] =  ALP1 * 3.0 * Y * D / R5
        DU[ 9] = -DU[5]
        DU[10] = -DU[8]
        DU[11] =  ALP1 * C3 / R3
        
    for I in range(N_variable):
        U[I] = U[I] + POT4 / PI2 * DU[I]


    return U



def _UB0(X, Y, D, Z, POT1, POT2, POT3, POT4, C0, C1, compute_strain):
    """
    Displacement and strain at depth (Part-B) 
    due to buried point source in a semi-infinite medium.

    Parameters
    ----------
    X, Y, D, Z : torch.Tensor
        Station coordinates in fault system.
    POT1, POT2, POT3, POT4 : float or torch.Tensor
        Strike-, dip-, tensile- and inflate-potency.
    C0, C1
        `MediumConstants` and `PointGeometry` bundles.
    compute_strain : bool
        Option to calculate the spatial derivative of the displacement.
        New in the PyTorch implementation.

    Returns
    -------
    U : list of torch.Tensor
        If `compute_strain` is `True`, 
        U is a list of 3 displacements and 9 spatial derivatives:
        [UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]
        If `False`, U is a list of 3 displacements only:
        [UX, UY, UZ]
    """

    # Initialization
    N_variable = 12 if compute_strain else 3
    U = [torch.zeros_like(X) for _ in range(N_variable)]
    DU = [None] * N_variable

    ALP3, SD, SDSD, SDCD = C0.ALP3, C0.SD, C0.SDSD, C0.SDCD
    P, Q, XY, X2, Y2 = C1.P, C1.Q, C1.XY, C1.X2, C1.Y2
    R, R2, R3, QR = C1.R, C1.R2, C1.R3, C1.QR
    
    C = D + Z
    RD = R + D
    D12 = 1.0 / (R * RD**2)
    D32 = D12 * (2.0 * R + D) / R2
    D33 = D12 * (3.0 * R + D) / (R2 * RD)

    FI1 = Y * (D12 - X2 * D33)
    FI2 = X * (D12 - Y2 * D33)
    FI3 = X / R3 - FI2
    FI4 = -XY * D32
    FI5 = 1.0 / (R * RD) - X2 * D32


    if compute_strain:
        D2, R5, QRX, A3, A5, B3, C3 = C1.D2, C1.R5, C1.QRX, C1.A3, C1.A5, C1.B3, C1.C3 
        UY, VY, WY, UZ, VZ, WZ = C1.UY, C1.VY, C1.WY, C1.UZ, C1.VZ, C1.WZ
        D53 = D12 * (8.0 * R2 + 9.0 * R * D + 3.0 * D2) / (R2**2 * RD)
        D54 = D12 * (5.0 * R2 + 4.0 * R * D + D2) / R3 * D12
        FJ1 = -3.0 * XY * (D33 - X2 * D54)
        FJ2 = 1.0 / R3 - 3.0 * D12 + 3.0 * X2 * Y2 * D54
        FJ3 = A3 / R3 - FJ2
        FJ4 = -3.0 * XY / R5 - FJ1
        FK1 = -Y * (D32 - X2 * D53)
        FK2 = -X * (D32 - Y2 * D53)
        FK3 = -3.0 * X * D / R5 - FK2



    # STRIKE-SLIP CONTRIBUTION
    DU[ 0] = -X2 * QR    - ALP3 * FI1 * SD
    DU[ 1] = -XY * QR    - ALP3 * FI2 * SD
    DU[ 2] = -C * X * QR - ALP3 * FI4 * SD

    if compute_strain:
        DU[ 3] = -X * QR * (1.0 + A5)         - ALP3 * FJ1 * SD
        DU[ 4] = -Y * QR * A5                 - ALP3 * FJ2 * SD
        DU[ 5] = -C * QR * A5                 - ALP3 * FK1 * SD
        DU[ 6] = -3.0 * X2 / R5 * UY          - ALP3 * FJ2 * SD
        DU[ 7] = -3.0 * XY / R5 * UY - X * QR - ALP3 * FJ4 * SD
        DU[ 8] = -3.0 * C * X / R5 * UY       - ALP3 * FK2 * SD
        DU[ 9] = -3.0 * X2 / R5 * UZ          + ALP3 * FK1 * SD
        DU[10] = -3.0 * XY / R5 * UZ          + ALP3 * FK2 * SD
        DU[11] =  3.0 * X / R5 * (-C * UZ + ALP3 * Y * SD)

    for I in range(N_variable):
        U[I] = U[I] + POT1 / PI2 * DU[I]


    # DIP-SLIP CONTRIBUTION
    DU[ 0] = -X * P * QR + ALP3 * FI3 * SDCD
    DU[ 1] = -Y * P * QR + ALP3 * FI1 * SDCD
    DU[ 2] = -C * P * QR + ALP3 * FI5 * SDCD

    if compute_strain:
        DU[ 3] = -P * QR * A5                + ALP3 * FJ3 * SDCD
        DU[ 4] =  Y * P * QRX                + ALP3 * FJ1 * SDCD
        DU[ 5] =  C * P * QRX                + ALP3 * FK3 * SDCD
        DU[ 6] = -3.0 * X / R5 * VY          + ALP3 * FJ1 * SDCD
        DU[ 7] = -3.0 * Y / R5 * VY - P * QR + ALP3 * FJ2 * SDCD
        DU[ 8] = -3.0 * C / R5 * VY          + ALP3 * FK1 * SDCD
        DU[ 9] = -3.0 * X / R5 * VZ          - ALP3 * FK3 * SDCD
        DU[10] = -3.0 * Y / R5 * VZ          - ALP3 * FK1 * SDCD
        DU[11] = -3.0 * C / R5 * VZ          + ALP3 * A3 / R3 * SDCD

    for I in range(N_variable):
        U[I] = U[I] + POT2 / PI2 * DU[I]


    # TENSILE-FAULT CONTRIBUTION
    DU[ 0] = X * Q * QR - ALP3 * FI3 * SDSD
    DU[ 1] = Y * Q * QR - ALP3 * FI1 * SDSD
    DU[ 2] = C * Q * QR - ALP3 * FI5 * SDSD

    if compute_strain:
        DU[ 3] =  Q * QR * A5       - ALP3 * FJ3 * SDSD
        DU[ 4] = -Y * Q * QRX       - ALP3 * FJ1 * SDSD
        DU[ 5] = -C * Q * QRX       - ALP3 * FK3 * SDSD
        DU[ 6] =  X * QR * WY       - ALP3 * FJ1 * SDSD
        DU[ 7] =  QR * (Y * WY + Q) - ALP3 * FJ2 * SDSD
        DU[ 8] =  C * QR * WY       - ALP3 * FK1 * SDSD
        DU[ 9] =  X * QR * WZ       + ALP3 * FK3 * SDSD
        DU[10] =  Y * QR * WZ       + ALP3 * FK1 * SDSD
        DU[11] =  C * QR * WZ       - ALP3 * A3 / R3 * SDSD

    for I in range(N_variable):
        U[I] = U[I] + POT3 / PI2 * DU[I]


    # INFLATE SOURCE CONTRIBUTION
    DU[ 0] = ALP3 * X / R3
    DU[ 1] = ALP3 * Y / R3
    DU[ 2] = ALP3 * D / R3

    if compute_strain:
        DU[ 3] =  ALP3 * A3 / R3
        DU[ 4] = -ALP3 * 3.0 * XY / R5
        DU[ 5] = -ALP3 * 3.0 * X * D / R5
        DU[ 6] =  DU[4]
        DU[ 7] =  ALP3 * B3 / R3
        DU[ 8] = -ALP3 * 3.0 * Y * D / R5
        DU[ 9] = -DU[5]
        DU[10] = -DU[8]
        DU[11] = -ALP3 * C3 / R3

    for I in range(N_variable):
        U[I] = U[I] + POT4 / PI2 * DU[I]


    return U



def _UC0(X, Y, D, Z, POT1, POT2, POT3, POT4, C0, C1, compute_strain):
    """
    Displacement and strain at depth (Part-C) 
    due to buried point source in a semi-infinite medium.

    Parameters
    ----------
    X, Y, D, Z : torch.Tensor
        Station coordinates in fault system.
    POT1, POT2, POT3, POT4 : float or torch.Tensor
        Strike-, dip-, tensile- and inflate-potency.
    C0, C1
        `MediumConstants` and `PointGeometry` bundles.
    compute_strain : bool
        Option to calculate the spatial derivative of the displacement.
        New in the PyTorch implementation.

    Returns
    -------
    U : list of torch.Tensor
        If `compute_strain` is `True`, 
        U is a list of 3 displacements and 9 spatial derivatives:
        [UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]
        If `False`, U is a list of 3 displacements only:
        [UX, UY, UZ]
    """

    # Initialization
    N_variable = 12 if compute_strain else 3
    U = [torch.zeros_like(X) for _ in range(N_variable)]
    DU = [None] * N_variable


    ALP4, ALP5, SD, CD, SDSD, SDCD, S2D, C2D = C0.ALP4, C0.ALP5, C0.SD, C0.CD, C0.SDSD, C0.SDCD, C0.S2D, C0.C2D
    P, Q, S, T = C1.P, C1.Q, C1.S, C1.T
    R2, R3, R5, R7, QR, QRX, A3, A5, C3 = C1.R2, C1.R3, C1.R5, C1.R7, C1.QR, C1.QRX, C1.A3, C1.A5, C1.C3 

    C = D + Z
    QR5 = 5.0 * Q / R2


    if compute_strain:
        XY, X2, Y2, D2 = C1.XY, C1.X2, C1.Y2, C1.D2
        Q2 = Q**2
        A7 = 1.0 - 7.0 * X2 / R2
        B5 = 1.0 - 5.0 * Y2 / R2
        B7 = 1.0 - 7.0 * Y2 / R2
        C5 = 1.0 - 5.0 * D2 / R2
        C7 = 1.0 - 7.0 * D2 / R2
        D7 = 2.0 - 7.0 * Q2 / R2
        QR7 = 7.0 * Q / R2
        DR5 = 5.0 * D / R2


    # STRIKE-SLIP CONTRIBUTION
    DU[ 0] = -ALP4 * A3 / R3 * CD + ALP5 * C * QR * A5
    DU[ 1] = 3.0 * X / R5 * ( ALP4 * Y * CD + ALP5 * C * (SD - Y * QR5))
    DU[ 2] = 3.0 * X / R5 * (-ALP4 * Y * SD + ALP5 * C * (CD + D * QR5))

    if compute_strain:
        DU[ 3] = ALP4 * 3.0 * X / R5 * (2.0 + A5) * CD - ALP5 * C * QRX * (2.0 + A7)
        DU[ 4] = 3.0 / R5 * ( ALP4 * Y * A5 * CD + ALP5 * C * (A5 * SD - Y * QR5 * A7))
        DU[ 5] = 3.0 / R5 * (-ALP4 * Y * A5 * SD + ALP5 * C * (A5 * CD + D * QR5 * A7))
        DU[ 6] = DU[4]
        DU[ 7] = 3.0 * X / R5 * ( ALP4 * B5 * CD - ALP5 * 5.0 * C / R2 * (2.0 * Y * SD + Q * B7))
        DU[ 8] = 3.0 * X / R5 * (-ALP4 * B5 * SD + ALP5 * 5.0 * C / R2 * (D * B7 * SD - Y * C7 * CD))
        DU[ 9] = 3.0 / R5      * (-ALP4 * D * A5 * CD + ALP5 * C * (A5 * CD + D * QR5 * A7))
        DU[10] = 15.0 * X / R7 * ( ALP4 * Y * D * CD  + ALP5 * C * (D * B7 * SD - Y * C7 * CD))
        DU[11] = 15.0 * X / R7 * (-ALP4 * Y * D * SD  + ALP5 * C * (2.0* D * CD - Q * C7))

    for I in range(N_variable):
        U[I] = U[I] + POT1 / PI2 * DU[I]


    # DIP-SLIP CONTRIBUTION
    DU[ 0] =  ALP4 * 3.0 * X * T / R5              - ALP5 * C * P * QRX
    DU[ 1] = -ALP4 / R3 * (C2D - 3.0 * Y * T / R2) + ALP5 * 3.0 * C / R5 * (S - Y * P * QR5)
    DU[ 2] = -ALP4 * A3 / R3 * SDCD                + ALP5 * 3.0 * C / R5 * (T + D * P * QR5)

    if compute_strain:
        DU[ 3] = ALP4 * 3.0 * T / R5 * A5                        - ALP5 * 5.0 * C * P * QR / R2 * A7
        DU[ 4] = 3.0 * X / R5 * (ALP4 * (C2D - 5.0 * Y * T / R2) - ALP5 * 5.0 * C / R2 * (S - Y * P * QR7))
        DU[ 5] = 3.0 * X / R5 * (ALP4 * (2.0 + A5) * SDCD        - ALP5 * 5.0 * C / R2 * (T + D * P * QR7))
        DU[ 6] = DU[4]
        DU[ 7] = 3.0 / R5 *     ( ALP4 * (2.0 * Y * C2D + T * B5)      + ALP5 * C * (S2D - 10.0 * Y * S / R2 - P * QR5 * B7))
        DU[ 8] = 3.0 / R5 *     ( ALP4 * Y * A5 * SDCD                 - ALP5 * C * ((3.0 + A5) * C2D + Y * P * DR5 * QR7))
        DU[ 9] = 3.0 * X / R5 * (-ALP4 * (S2D - T * DR5)               - ALP5 * 5.0 * C / R2 * (T + D * P * QR7))
        DU[10] = 3.0 / R5 *     (-ALP4 * (D * B5 * C2D + Y * C5 * S2D) - ALP5 * C * ((3.0 + A5) * C2D + Y * P * DR5 * QR7))
        DU[11] = 3.0 / R5 *     (-ALP4 * D * A5 * SDCD                 - ALP5 * C * (S2D - 10.0 * D * T / R2 + P * QR5 * C7))

    for I in range(N_variable):
        U[I] = U[I] + POT2 / PI2 * DU[I]


    # TENSILE-FAULT CONTRIBUTION
    DU[ 0] = 3.0 * X / R5 * (-ALP4 * S + ALP5 * (C * Q * QR5 - Z))
    DU[ 1] =  ALP4 / R3 * (S2D - 3.0 * Y * S / R2) + ALP5 * 3.0 / R5 * (C * (T - Y + Y * Q * QR5) - Y * Z)
    DU[ 2] = -ALP4 / R3 * (1.0 - A3 * SDSD)        - ALP5 * 3.0 / R5 * (C * (S - D + D * Q * QR5) - D * Z)

    if compute_strain:
        DU[ 3] = -ALP4 * 3.0 * S / R5 * A5 + ALP5 * (C * QR * QR5 * A7 - 3.0 * Z / R5 * A5)
        DU[ 4] = 3.0 * X / R5 * (-ALP4 * (S2D - 5.0 * Y * S / R2)     - ALP5 * 5.0 / R2 * (C * (T - Y + Y * Q * QR7) - Y * Z))
        DU[ 5] = 3.0 * X / R5 * ( ALP4 * (1.0 - (2.0 + A5) * SDSD)    + ALP5 * 5.0 / R2 * (C * (S - D + D * Q * QR7) - D * Z))
        DU[ 6] = DU[4]
        DU[ 7] = 3.0 / R5 *     (-ALP4 * (2.0 * Y * S2D + S * B5)     - ALP5 * (C * (2.0 * SDSD + 10.0 * Y * (T - Y) / R2 - Q * QR5 * B7) + Z * B5))
        DU[ 8] = 3.0 / R5 *     ( ALP4 * Y * (1.0 - A5 * SDSD)        + ALP5 * (C * (3.0 + A5) * S2D - Y * DR5 * (C * D7 + Z)))
        DU[ 9] = 3.0 * X / R5 * (-ALP4 * (C2D+ S * DR5)               + ALP5 * (5.0 * C / R2 * (S - D + D * Q *QR7) - 1.0 - Z * DR5))
        DU[10] = 3.0 / R5 *     ( ALP4 * (D * B5 * S2D - Y* C5 * C2D) + ALP5 * (C * ((3.0 + A5) * S2D - Y * DR5 * D7) - Y * (1.0 + Z * DR5)))
        DU[11] = 3.0 / R5 *     (-ALP4 * D * (1.0 - A5 * SDSD)        - ALP5 * (C * (C2D + 10.0 * D * (S - D) / R2 - Q * QR5 * C7) + Z * (1.0 + C5)))

    for I in range(N_variable):
        U[I] = U[I] + POT3 / PI2 * DU[I]


    # INFLATE SOURCE CONTRIBUTION
    DU[ 0] = ALP4 * 3.0 * X * D / R5
    DU[ 1] = ALP4 * 3.0 * Y * D / R5
    DU[ 2] = ALP4 * C3 / R3

    if compute_strain:
        DU[ 3] =  ALP4 * 3.0 * D / R5 * A5
        DU[ 4] = -ALP4 * 15.0 * XY * D / R7
        DU[ 5] = -ALP4 * 3.0 * X / R5 * C5
        DU[ 6] = DU[4]
        DU[ 7] =  ALP4 * 3.0 * D / R5 * B5
        DU[ 8] = -ALP4 * 3.0 * Y / R5 * C5
        DU[ 9] = DU[5]
        DU[10] = DU[8]
        DU[11] =  ALP4 * 3.0 * D / R5 * (2.0 + C5)
    
    for I in range(N_variable):
        U[I] = U[I] + POT4 / PI2 * DU[I]


    return U






def _UA(XI, ET, Q, DISL1, DISL2, DISL3, C0, C2, compute_strain):
    """
    Displacement and strain at depth (Part-A) 
    due to buried finite fault in a semi-infinite medium.

    Parameters
    ----------
    XI, ET, Q : torch.Tensor
        Station coordinates in fault system.
    DISL1, DISL2, DISL3 : float or torch.Tensor
        Strike-, dip-, tensile-dislocations.
    C0, C2
        `MediumConstants` and `FaultGeometry` bundles.
    compute_strain : bool
        Option to calculate the spatial derivative of the displacement.
        New in the PyTorch implementation.

    Returns
    -------
    U : list of torch.Tensor
        If `compute_strain` is `True`, 
        U is a list of 3 displacements and 9 spatial derivatives:
        [UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]
        If `False`, U is a list of 3 displacements only:
        [UX, UY, UZ]
    """

    # Initialization
    N_variable = 12 if compute_strain else 3
    U = [torch.zeros_like(XI) for _ in range(N_variable)]
    DU = [None] * N_variable

    ALP1, ALP2 = C0.ALP1, C0.ALP2
    R, TT, ALX, ALE, X11, Y11 = C2.R, C2.TT, C2.ALX, C2.ALE, C2.X11, C2.Y11
    QX = Q * X11
    QY = Q * Y11


    if compute_strain:
        SD, CD = C0.SD, C0.CD
        XI2, Q2, R3, Y, D, Y32 = C2.XI2, C2.Q2, C2.R3, C2.Y, C2.D, C2.Y32
        EY, EZ, FY, FZ, GY, GZ, HY, HZ = C2.EY, C2.EZ, C2.FY, C2.FZ, C2.GY, C2.GZ, C2.HY, C2.HZ
        XY = XI * Y11


    # STRIKE-SLIP CONTRIBUTION
    DU[ 0] = TT / 2.0   + ALP2 * XI * QY
    DU[ 1] =              ALP2 * Q / R
    DU[ 2] = ALP1 * ALE - ALP2 * Q * QY

    if compute_strain:
        DU[ 3] = -ALP1 * QY                 - ALP2 * XI2 * Q * Y32
        DU[ 4] =                            - ALP2 * XI * Q / R3
        DU[ 5] =  ALP1 * XY                 + ALP2 * XI * Q2 * Y32
        DU[ 6] =  ALP1 * XY * SD            + ALP2 * XI * FY + D / 2.0 * X11
        DU[ 7] =                              ALP2 * EY
        DU[ 8] =  ALP1 * (CD / R + QY * SD) - ALP2 * Q * FY
        DU[ 9] =  ALP1 * XY * CD            + ALP2 * XI * FZ + Y / 2.0 * X11
        DU[10] =                              ALP2 * EZ
        DU[11] = -ALP1 * (SD / R - QY * CD) - ALP2 * Q * FZ

    for I in range(N_variable):
        U[I] = U[I] + DISL1 / PI2 * DU[I]
    

    # DIP-SLIP CONTRIBUTION
    DU[ 0] =              ALP2 * Q / R
    DU[ 1] = TT / 2.0   + ALP2 * ET * QX
    DU[ 2] = ALP1 * ALX - ALP2 * Q * QX

    if compute_strain:
        DU[ 3] =                                 - ALP2 * XI * Q / R3
        DU[ 4] =  -QY / 2.0                      - ALP2 * ET * Q / R3
        DU[ 5] =  ALP1 / R                       + ALP2 * Q2 / R3
        DU[ 6] =                                   ALP2 * EY
        DU[ 7] =  ALP1 * D * X11 + XY / 2.0 * SD + ALP2 * ET * GY
        DU[ 8] =  ALP1 * Y * X11                 - ALP2 * Q * GY
        DU[ 9] =                                   ALP2 * EZ
        DU[10] =  ALP1 * Y * X11 + XY / 2.0 * CD + ALP2 * ET * GZ
        DU[11] = -ALP1 * D * X11                 - ALP2 * Q * GZ

    for I in range(N_variable):
        U[I] = U[I] + DISL2 / PI2 * DU[I]

    
    # TENSILE-FAULT CONTRIBUTION
    DU[ 0] = -ALP1 * ALE - ALP2 * Q * QY
    DU[ 1] = -ALP1 * ALX - ALP2 * Q * QX
    DU[ 2] =  TT / 2.0   - ALP2 * (ET * QX + XI * QY)

    if compute_strain:
        DU[ 3] = -ALP1 * XY                  + ALP2 * XI * Q2 * Y32
        DU[ 4] = -ALP1 / R                   + ALP2 * Q2 / R3
        DU[ 5] = -ALP1 * QY                  - ALP2 * Q * Q2 * Y32
        DU[ 6] = -ALP1 * (CD / R + QY * SD)  - ALP2 * Q * FY
        DU[ 7] = -ALP1 * Y * X11             - ALP2 * Q * GY
        DU[ 8] =  ALP1 * (D * X11 + XY * SD) + ALP2 * Q * HY
        DU[ 9] =  ALP1 * (SD / R - QY * CD)  - ALP2 * Q * FZ
        DU[10] =  ALP1 * D * X11             - ALP2 * Q * GZ
        DU[11] =  ALP1 * (Y * X11 + XY * CD) + ALP2 * Q * HZ

    for I in range(N_variable):
        U[I] = U[I] + DISL3 / PI2 * DU[I]


    return U



def _UB(XI, ET, Q, DISL1, DISL2, DISL3, C0, C2, compute_strain):
    """
    Displacement and strain at depth (Part-B) 
    due to buried finite fault in a semi-infinite medium.

    Parameters
    ----------
    XI, ET, Q : torch.Tensor
        Station coordinates in fault system.
    DISL1, DISL2, DISL3 : float or torch.Tensor
        Strike-, dip-, tensile-dislocations.
    C0, C2
        `MediumConstants` and `FaultGeometry` bundles.
    compute_strain : bool
        Option to calculate the spatial derivative of the displacement.
        New in the PyTorch implementation.

    Returns
    -------
    U : list of torch.Tensor
        If `compute_strain` is `True`, 
        U is a list of 3 displacements and 9 spatial derivatives:
        [UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]
        If `False`, U is a list of 3 displacements only:
        [UX, UY, UZ]
    """

    # Initialization
    N_variable = 12 if compute_strain else 3
    U = [torch.zeros_like(XI) for _ in range(N_variable)]
    DU = [None] * N_variable

    ALP3, SD, CD, SDSD, CDCD, SDCD = C0.ALP3, C0.SD, C0.CD, C0.SDSD, C0.CDCD, C0.SDCD
    XI2, Q2, R, Y, D, TT = C2.XI2, C2.Q2, C2.R, C2.Y, C2.D, C2.TT
    ALE, X11, Y11 = C2.ALE, C2.X11, C2.Y11
        
    RD = R + D
    RD2 = RD**2

    # See the corresponding block in `_SRECTG`: both formulae are evaluated and
    # then selected, with CD_safe == 1 wherever the vertical branch wins.
    SD = _as_tensor_like(SD, R)
    CD = _as_tensor_like(CD, R)
    vertical = CD == 0.0
    CD_safe = torch.where(vertical, torch.ones_like(CD), CD)
    CDCD_safe = CD_safe**2
    xi_nonzero = XI != 0.0

    X = torch.sqrt(XI2 + Q2)
    AI3_inclined, AI4_inclined = _ub_AI_inclined(
        XI, ET, Q, SD, CD_safe, SDCD, R, X, RD, Y, ALE, xi_nonzero)

    AI3_vertical = (ET / RD + Y * Q / RD2 - ALE) / 2.0
    AI4_vertical = XI * Y / RD2 / 2.0

    AI3_value = torch.where(vertical, AI3_vertical, AI3_inclined)
    AI4_value = torch.where(vertical, AI4_vertical, AI4_inclined)

    # Same surrogate-gradient treatment as in `_SRECTG`.
    ill_conditioned = torch.abs(CD) < DIP_GRAD_FLOOR
    SD_off, CD_off = _dip_offset(SD, CD)
    D_off  = ET * SD_off - Q * CD_off
    Y_off  = ET * CD_off + Q * SD_off
    RD_off = R + D_off
    AI3_surr, AI4_surr = _ub_AI_inclined(
        XI, ET, Q, SD_off, CD_off, SD_off * CD_off, R, X, RD_off, Y_off, ALE,
        xi_nonzero)

    AI3 = _surrogate_grad(ill_conditioned, AI3_value, AI3_surr)
    AI4 = _surrogate_grad(ill_conditioned, AI4_value, AI4_surr)

    AI1 = -XI / RD * CD - AI4 * SD
    AI2 = torch.log(RD) + AI3 * SD
    QX = Q * X11
    QY = Q * Y11


    if compute_strain:
        R3, Y32 = C2.R3, C2.Y32
        EY, EZ, FY, FZ, GY, GZ, HY, HZ = C2.EY, C2.EZ, C2.FY, C2.FZ, C2.GY, C2.GZ, C2.HY, C2.HZ

        D11 = 1.0 / (R * RD)
        AJ2 = XI * Y / RD * D11
        AJ5 = -(D + Y**2 / RD) * D11

        AK1_inclined, AK3_inclined, AJ3_inclined, AJ6_inclined = _ub_AJK_inclined(
            XI, Q, SD, CD_safe, Y, Y11, D11, AJ2, AJ5)

        AK1_vertical = XI * Q / RD * D11
        AK3_vertical = SD / RD * (XI2 * D11 - 1.0)
        AJ3_vertical = -XI / RD2 * (Q2 * D11 - 0.5)
        AJ6_vertical = - Y / RD2 * (XI2 * D11 - 0.5)

        AK1_value = torch.where(vertical, AK1_vertical, AK1_inclined)
        AK3_value = torch.where(vertical, AK3_vertical, AK3_inclined)
        AJ3_value = torch.where(vertical, AJ3_vertical, AJ3_inclined)
        AJ6_value = torch.where(vertical, AJ6_vertical, AJ6_inclined)

        D11_off = 1.0 / (R * RD_off)
        AJ2_off = XI * Y_off / RD_off * D11_off
        AJ5_off = -(D_off + Y_off**2 / RD_off) * D11_off
        AK1_surr, AK3_surr, AJ3_surr, AJ6_surr = _ub_AJK_inclined(
            XI, Q, SD_off, CD_off, Y_off, Y11, D11_off, AJ2_off, AJ5_off)

        AK1 = _surrogate_grad(ill_conditioned, AK1_value, AK1_surr)
        AK3 = _surrogate_grad(ill_conditioned, AK3_value, AK3_surr)
        AJ3 = _surrogate_grad(ill_conditioned, AJ3_value, AJ3_surr)
        AJ6 = _surrogate_grad(ill_conditioned, AJ6_value, AJ6_surr)

        XY = XI * Y11
        AK2 = 1.0 / R + AK3 * SD
        AK4 = XY * CD - AK1 * SD
        AJ1 = AJ5 * CD - AJ6 * SD
        AJ4 = -XY - AJ2 * CD + AJ3 * SD



    # STRIKE-SLIP CONTRIBUTION
    DU[ 0] = -XI * QY - TT - ALP3 * AI1 * SD
    DU[ 1] = -Q / R        + ALP3 * Y / RD * SD
    DU[ 2] =  Q * QY       - ALP3 * AI2 * SD

    if compute_strain:
        DU[ 3] =  XI2 * Q * Y32     - ALP3 * AJ1 * SD
        DU[ 4] =  XI * Q / R3       - ALP3 * AJ2 * SD
        DU[ 5] = -XI * Q2 * Y32     - ALP3 * AJ3 * SD
        DU[ 6] = -XI * FY - D * X11 + ALP3 * (XY + AJ4) * SD
        DU[ 7] = -EY                + ALP3 * (1.0 / R + AJ5) * SD
        DU[ 8] =  Q * FY            - ALP3 * (QY - AJ6) * SD
        DU[ 9] = -XI * FZ - Y * X11 + ALP3 * AK1 * SD
        DU[10] = -EZ                + ALP3 * Y * D11 * SD
        DU[11] =  Q*FZ              + ALP3 * AK2 * SD

    for I in range(N_variable):
        U[I] = U[I] + DISL1 / PI2 * DU[I]


    # DIP-SLIP CONTRIBUTION
    DU[ 0] = -Q / R        + ALP3 * AI3 * SDCD
    DU[ 1] = -ET * QX - TT - ALP3 * XI / RD * SDCD
    DU[ 2] =  Q * QX       + ALP3 * AI4 * SDCD

    if compute_strain:
        DU[ 3] =  XI * Q / R3       + ALP3 * AJ4 * SDCD
        DU[ 4] =  ET * Q / R3 + QY  + ALP3 * AJ5 * SDCD
        DU[ 5] = -Q2 / R3           + ALP3 * AJ6 * SDCD
        DU[ 6] = -EY                + ALP3 * AJ1 * SDCD
        DU[ 7] = -ET * GY - XY * SD + ALP3 * AJ2 * SDCD
        DU[ 8] =  Q*GY              + ALP3 * AJ3 * SDCD
        DU[ 9] = -EZ                - ALP3 * AK3 * SDCD
        DU[10] = -ET * GZ - XY * CD - ALP3 * XI * D11 * SDCD
        DU[11] =  Q * GZ            - ALP3 * AK4 * SDCD

    for I in range(N_variable):
        U[I] = U[I] + DISL2 / PI2 * DU[I]


    # TENSILE-FAULT CONTRIBUTION
    DU[ 0] = Q * QY                 - ALP3 * AI3 * SDSD
    DU[ 1] = Q * QX                 + ALP3 * XI / RD * SDSD
    DU[ 2] = ET * QX + XI * QY - TT - ALP3 * AI4 * SDSD

    if compute_strain:
        DU[ 3] = -XI * Q2 * Y32 - ALP3 * AJ4 * SDSD
        DU[ 4] = -Q2 / R3       - ALP3 * AJ5 * SDSD
        DU[ 5] =  Q * Q2 * Y32  - ALP3 * AJ6 * SDSD
        DU[ 6] =  Q * FY        - ALP3 * AJ1 * SDSD
        DU[ 7] =  Q * GY        - ALP3 * AJ2 * SDSD
        DU[ 8] = -Q * HY        - ALP3 * AJ3 * SDSD
        DU[ 9] =  Q * FZ        + ALP3 * AK3 * SDSD
        DU[10] =  Q * GZ        + ALP3 * XI * D11 * SDSD
        DU[11] = -Q * HZ        + ALP3 * AK4 * SDSD

    for I in range(N_variable):
        U[I] = U[I] + DISL3 / PI2 * DU[I]


    return U



def _UC(XI, ET, Q, Z, DISL1, DISL2, DISL3, C0, C2, compute_strain):
    """
    Displacement and strain at depth (Part-C) 
    due to buried finite fault in a semi-infinite medium.

    Parameters
    ----------
    XI, ET, Q, Z : torch.Tensor
        Station coordinates in fault system.
    DISL1, DISL2, DISL3 : float or torch.Tensor
        Strike-, dip-, tensile-dislocations.
    C0, C2
        `MediumConstants` and `FaultGeometry` bundles.
    compute_strain : bool
        Option to calculate the spatial derivative of the displacement.
        New in the PyTorch implementation.

    Returns
    -------
    U : list of torch.Tensor
        If `compute_strain` is `True`, 
        U is a list of 3 displacements and 9 spatial derivatives:
        [UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]
        If `False`, U is a list of 3 displacements only:
        [UX, UY, UZ]
    """

    # Initialization
    N_variable = 12 if compute_strain else 3
    U = [torch.zeros_like(XI) for _ in range(N_variable)]
    DU = [None] * N_variable

    ALP4, ALP5, SD, CD  = C0.ALP4, C0.ALP5, C0.SD, C0.CD, 
    XI2, Q2, R, R3, Y, D = C2.XI2, C2.Q2, C2.R, C2.R3, C2.Y, C2.D
    X11, Y11, X32, Y32 = C2.X11, C2.Y11, C2.X32, C2.Y32

    C = D + Z 
    H = Q * CD - Z
    Z32 = SD / R3 - H * Y32
    XY = XI * Y11
    QY = Q * Y11


    if compute_strain:
        SDSD, CDCD, SDCD = C0.SDSD, C0.CDCD, C0.SDCD
        ET2, R2, R5 = C2.ET2, C2.R2, C2.R5
        X53 = (8.0 * R2 + 9.0 * R * XI + 3.0 * XI2) * X11**3 / R2
        Y53 = (8.0 * R2 + 9.0 * R * ET + 3.0 * ET2) * Y11**3 / R2
        Z53 = 3.0 * SD / R5 - H * Y53
        Y0 = Y11 - XI2 * Y32
        Z0 = Z32 - XI2 * Z53
        PPY = CD / R3 + Q * Y32 * SD
        PPZ = SD / R3 - Q * Y32 * CD
        QQ = Z * Y32 + Z32 + Z0
        QQY = 3.0 * C * D / R5 - QQ * SD
        QQZ = 3.0 * C * Y / R5 - QQ * CD + Q * Y32
        QR = 3.0 * Q / R5
        CDR = (C + D) / R3
        YY0 = Y / R3 - Y0 * CD



    # STRIKE-SLIP CONTRIBUTION
    DU[ 0] = ALP4 * XY * CD                  - ALP5 * XI * Q * Z32
    DU[ 1] = ALP4 * (CD / R + 2.0 * QY * SD) - ALP5 * C * Q / R3
    DU[ 2] = ALP4 * QY * CD                  - ALP5 * (C * ET / R3 - Z * Y11 + XI2 * Z32)

    if compute_strain:
        DU[ 3] =  ALP4 * Y0 * CD                                     - ALP5 * Q * Z0
        DU[ 4] = -ALP4 * XI * (CD / R3 + 2.0 * Q * Y32 * SD)         + ALP5 * C * XI * QR
        DU[ 5] = -ALP4 * XI * Q * Y32 * CD                           + ALP5 * XI * (3.0 * C * ET / R5 - QQ)
        DU[ 6] = -ALP4 * XI * PPY * CD                               - ALP5 * XI * QQY
        DU[ 7] =  ALP4 * 2.0 * (D / R3 - Y0 * SD) * SD - Y / R3 * CD - ALP5 * (CDR * SD - ET / R3 - C * Y * QR)
        DU[ 8] = -ALP4 * Q / R3 + YY0 * SD                           + ALP5 * (CDR * CD + C * D * QR - (Y0 * CD + Q * Z0) * SD)
        DU[ 9] =  ALP4 * XI * PPZ * CD                               - ALP5 * XI * QQZ
        DU[10] =  ALP4 * 2.0 * (Y / R3 - Y0 * CD) * SD + D / R3 * CD - ALP5 * (CDR * CD + C * D * QR)
        DU[11] =  YY0 * CD                                           - ALP5 * (CDR * SD - C * Y * QR - Y0 * SDSD + Q * Z0 * CD)

    for I in range(N_variable):
        U[I] = U[I] + DISL1 / PI2 * DU[I]


    # DIP-SLIP CONTRIBUTION
    DU[ 0] =  ALP4 * CD / R - QY * SD - ALP5 * C * Q / R3
    DU[ 1] =  ALP4 * Y * X11          - ALP5 * C * ET * Q * X32
    DU[ 2] = -D * X11 - XY * SD       - ALP5 * C * (X11 - Q2 * X32)

    if compute_strain:
        DU[ 3] = -ALP4 * XI / R3 * CD              + ALP5 * C * XI * QR + XI * Q * Y32 * SD
        DU[ 4] = -ALP4 * Y / R3                    + ALP5 * C * ET * QR
        DU[ 5] =  D / R3 - Y0 * SD                 + ALP5 * C / R3 * (1.0 - 3.0 * Q2 / R2)
        DU[ 6] = -ALP4 * ET / R3 + Y0 * SDSD       - ALP5 * (CDR * SD - C * Y * QR)
        DU[ 7] =  ALP4 * (X11 - Y**2 * X32)        - ALP5 * C * ((D + 2.0 * Q * CD) * X32 - Y * ET * Q * X53)
        DU[ 8] =   XI * PPY * SD + Y * D * X32     + ALP5 * C * ((Y + 2.0 * Q * SD) * X32 - Y * Q2 * X53)
        DU[ 9] = -Q / R3 + Y0 * SDCD               - ALP5 * (CDR * CD + C * D * QR)
        DU[10] =  ALP4 * Y * D * X32               - ALP5 * C * ((Y - 2.0 * Q * SD) * X32 + D * ET * Q * X53)
        DU[11] = -XI * PPZ * SD + X11 - D**2 * X32 - ALP5 * C * ((D - 2.0 * Q * CD) * X32 - D * Q2 * X53)

    for I in range(N_variable):
        U[I] = U[I] + DISL2 / PI2 * DU[I]


    # TENSILE-FAULT CONTRIBUTION
    DU[ 0] = -ALP4 * (SD / R + QY * CD)      - ALP5 * (Z * Y11 - Q2 * Z32)
    DU[ 1] =  ALP4 * 2.0 * XY * SD + D * X11 - ALP5 * C * (X11 - Q2 * X32)
    DU[ 2] =  ALP4 * (Y * X11 + XY * CD)     + ALP5 * Q * (C * ET * X32 + XI * Z32)

    if compute_strain:
        DU[ 3] =  ALP4 * XI / R3 * SD + XI * Q * Y32 * CD       + ALP5 * XI * (3.0 * C * ET / R5 - 2.0 * Z32 - Z0)
        DU[ 4] =  ALP4 * 2.0 * Y0 * SD - D / R3                 + ALP5 * C / R3 * (1.0 - 3.0 * Q2 / R2)
        DU[ 5] = -ALP4 * YY0                                    - ALP5 * (C * ET * QR - Q * Z0)
        DU[ 6] =  ALP4 * (Q / R3 + Y0 * SDCD)                   + ALP5 * (Z / R3 * CD + C * D * QR - Q * Z0 * SD)
        DU[ 7] = -ALP4 * 2.0 * XI * PPY * SD - Y * D * X32      + ALP5 *  C * ((Y + 2.0 * Q * SD) * X32 - Y * Q2 * X53)
        DU[ 8] = -ALP4 * (XI * PPY * CD - X11 + Y**2 * X32)     + ALP5 * (C * ((D + 2.0 * Q * CD) * X32 - Y * ET * Q * X53) + XI * QQY)
        DU[ 9] = -ET / R3 + Y0 * CDCD                           - ALP5 * (Z / R3 * SD - C * Y * QR - Y0 * SDSD + Q * Z0 * CD)
        DU[10] =  ALP4 * 2.0 * XI * PPZ * SD - X11 + D**2 * X32 - ALP5 *  C * ((D - 2.0 * Q * CD) * X32 - D * Q2 * X53)
        DU[11] =  ALP4 * (XI * PPZ * CD + Y * D * X32)          + ALP5 * (C * ((Y - 2.0 * Q * SD) * X32 + D * ET * Q * X53) + XI * QQZ)

    for I in range(N_variable):
        U[I] = U[I] + DISL3 / PI2 * DU[I]


    return U





# ---------------------------------------------------------------------------
# Precomputed quantities shared between the parts of a kernel (the FORTRAN
# COMMON blocks).  The first port made these classes whose `DCCONn` method
# filled in `self`, so one instance was overwritten in place -- eight times per
# DC3D call -- while `_UA`/`_UB`/`_UC` read from it: nothing tied a reader to
# the values it expected, and the four corners could not be evaluated together.
# Named tuples instead, one fresh bundle per call.  Field access is unchanged,
# so the `_UAn`/`_UBn`/`_UCn` functions did not move.
# ---------------------------------------------------------------------------
class MediumConstants(NamedTuple):
    """Medium constants and fault-dip constants (FORTRAN COMMON /C0/)."""
    ALP1: Tensor
    ALP2: Tensor
    ALP3: Tensor
    ALP4: Tensor
    ALP5: Tensor
    SD: Tensor
    CD: Tensor
    SDSD: Tensor
    CDCD: Tensor
    SDCD: Tensor
    S2D: Tensor
    C2D: Tensor


def medium_constants(ALPHA, DIP, is_degree):
    """Medium constants and fault-dip constants.

    `ALPHA` is (lambda+myu)/(lambda+2*myu); `DIP` is in degrees if `is_degree`.
    """
    if is_degree:
        SD = torch.sin(torch.deg2rad(DIP))
        CD = torch.cos(torch.deg2rad(DIP))
    else:
        SD = torch.sin(DIP)
        CD = torch.cos(DIP)

    # Snap a near-vertical dip so the rectangular formulae take their
    # vertical-fault branch instead of dividing by a cosine of 6e-17.  On the
    # *value* only: a plain torch.where would sever d/d(dip) here, before the
    # kernel sees it, and nothing downstream could recover it.
    mask = (torch.abs(CD) < EPS_DIP)
    SD = _surrogate_grad(mask, torch.where(mask, torch.sign(SD), SD), SD)
    CD = _surrogate_grad(mask, torch.where(mask, torch.zeros_like(CD), CD), CD)

    SDSD, CDCD, SDCD = SD**2, CD**2, SD * CD
    return MediumConstants(
        ALP1=(1.0 - ALPHA) / 2.0,
        ALP2=ALPHA / 2.0,
        ALP3=(1.0 - ALPHA) / ALPHA,
        ALP4=1.0 - ALPHA,
        ALP5=ALPHA,
        SD=SD, CD=CD,
        SDSD=SDSD, CDCD=CDCD, SDCD=SDCD,
        S2D=2.0 * SDCD,
        C2D=CDCD - SDSD,
    )


class PointGeometry(NamedTuple):
    """Station geometry for a point source (FORTRAN COMMON /C1/)."""
    P: Tensor
    Q: Tensor
    S: Tensor
    T: Tensor
    XY: Tensor
    X2: Tensor
    Y2: Tensor
    D2: Tensor
    R: Tensor
    R2: Tensor
    R3: Tensor
    R5: Tensor
    R7: Tensor
    QR: Tensor
    QRX: Tensor
    A3: Tensor
    A5: Tensor
    B3: Tensor
    C3: Tensor
    UY: Tensor
    VY: Tensor
    WY: Tensor
    UZ: Tensor
    VZ: Tensor
    WZ: Tensor


def point_geometry(X, Y, D, c0):
    """Station geometry for a point source, at fault-frame coordinates (X, Y, D).

    Coordinates negligible next to their radius are snapped to zero.
    """
    SD, CD = c0.SD, c0.CD

    scale = torch.sqrt(X**2 + Y**2 + D**2)
    X, Y, D = (_snap(v, scale) for v in (X, Y, D))

    P = Y * CD + D * SD
    Q = Y * SD - D * CD
    X2, Y2, D2 = X**2, Y**2, D**2
    R2 = X2 + Y2 + D2
    R = torch.sqrt(R2)
    S = P * SD + Q * CD
    T = P * CD - Q * SD
    QR = 3.0 * Q / R**5
    UY = SD - 5.0 * Y * Q / R2
    UZ = CD + 5.0 * D * Q / R2
    return PointGeometry(
        P=P, Q=Q, S=S, T=T,
        XY=X * Y, X2=X2, Y2=Y2, D2=D2,
        R=R, R2=R2, R3=R**3, R5=R**5, R7=R**7,
        QR=QR, QRX=5.0 * QR * X / R2,
        A3=1.0 - 3.0 * X2 / R2,
        A5=1.0 - 5.0 * X2 / R2,
        B3=1.0 - 3.0 * Y2 / R2,
        C3=1.0 - 3.0 * D2 / R2,
        UY=UY, UZ=UZ,
        VY=S - 5.0 * Y * P * Q / R2,
        VZ=T + 5.0 * D * P * Q / R2,
        WY=UY + SD, WZ=UZ + CD,
    )


class FaultGeometry(NamedTuple):
    """Station geometry for a finite source (FORTRAN COMMON /C2/)."""
    XI2: Tensor
    ET2: Tensor
    Q2: Tensor
    R: Tensor
    R2: Tensor
    R3: Tensor
    R5: Tensor
    Y: Tensor
    D: Tensor
    TT: Tensor
    ALX: Tensor
    ALE: Tensor
    X11: Tensor
    Y11: Tensor
    X32: Tensor
    Y32: Tensor
    EY: Tensor
    EZ: Tensor
    FY: Tensor
    FZ: Tensor
    GY: Tensor
    GZ: Tensor
    HY: Tensor
    HZ: Tensor


def fault_geometry(XI, ET, Q, SD, CD, KXI, KET):
    """Station geometry for one corner of the rectangle.

    `KXI` / `KET` = 1 mark R+XI / R+ET as negligible next to R.  Coordinates
    negligible next to R are snapped to zero.
    """
    scale = torch.sqrt(XI**2 + ET**2 + Q**2)
    XI, ET, Q = (_snap(v, scale) for v in (XI, ET, Q))

    XI2, ET2, Q2 = XI**2, ET**2, Q**2
    R2 = XI2 + ET2 + Q2
    R = torch.sqrt(R2)
    R3 = R**3
    Y = ET * CD + Q * SD
    D = ET * SD - Q * CD

    # Denominators are sanitised before the division; see `_safe_div`.
    q_nonzero = Q != 0.0
    TT = torch.atan(_safe_div(XI * ET, Q * R, q_nonzero))

    RXI = R + XI
    kxi_regular = KXI != 1
    ALX = torch.where(kxi_regular,
                      _safe_log(RXI, kxi_regular),
                      -_safe_log(R - XI, ~kxi_regular))
    X11 = _safe_div(1.0, R * RXI, kxi_regular)
    X32 = torch.where(kxi_regular, (R + RXI) * X11**2 / R, torch.zeros_like(R))

    RET = R + ET
    ket_regular = KET != 1
    ALE = torch.where(ket_regular,
                      _safe_log(RET, ket_regular),
                      -_safe_log(R - ET, ~ket_regular))
    Y11 = _safe_div(1.0, R * RET, ket_regular)
    Y32 = torch.where(ket_regular, (R + RET) * Y11**2 / R, torch.zeros_like(R))

    return FaultGeometry(
        XI2=XI2, ET2=ET2, Q2=Q2,
        R=R, R2=R2, R3=R3, R5=R**5,
        Y=Y, D=D, TT=TT,
        ALX=ALX, ALE=ALE, X11=X11, Y11=Y11, X32=X32, Y32=Y32,
        EY=SD / R - Y * Q / R3,
        EZ=CD / R + D * Q / R3,
        FY=D / R3 + XI2 * Y32 * SD,
        FZ=Y / R3 + XI2 * Y32 * CD,
        GY=2.0 * X11 * SD - Y * Q * X32,
        GZ=2.0 * X11 * CD + D * Q * X32,
        HY=D * Q * X32 + XI * Q * Y32 * SD,
        HZ=Y * Q * X32 + XI * Q * Y32 * CD,
    )

