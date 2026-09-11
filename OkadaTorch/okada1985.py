import torch
from .utils import (PI2, _SRECTG, _blank_where, _dummy_where, _length_scale,
                    _on_fault_edge, _rel_eps)


def SPOINT(ALP, X, Y, D, SD, CD, DISL1, DISL2, DISL3,
           compute_strain=True, return_iret=False):
    """
    Surface displacement, strain, tilt due to buried point source 
    in a semi-infinite medium.

    Parameters
    ----------
    ALP : float or torch.Tensor
        Medium constant. myu/(lambda+myu)
    X, Y : torch.Tensor
        Coordinate of station.
    D : float or torch.Tensor
        Source depth.
    SD, CD : float or torch.Tensor
        Sine, cosine of dip-angle. 
        (CD=0.0, SD=+/-1.0 should be given for vertical fault.)
    DISL1, DISL2, DISL3 : float or torch.Tensor
        Strike-, dip- and tensile-dislocation.
    compute_strain : bool, default True
        Option to calculate the spatial derivative of the displacement. 
        New in the PyTorch implementation.
    return_iret : bool, default False
        Return the singularity flag. New in the PyTorch implementation.

    Returns
    -------
    U : list of torch.Tensor
        If `compute_strain` is `True`, 
        U is a list of 3 displacements and 6 spatial derivatives:
        [U1, U2, U3, U11, U12, U21, U22, U31, U32]
        If `False`, U is a list of 3 displacements only:
        [U1, U2, U3]
        Stations flagged by `IRET` are returned as exactly zero.

        U1, U2, U3 : torch.Tensor
            Displacement. unit = (unit of dislocation) / area
        U11, U12, U21, U22 : torch.Tensor
            Strain. unit = (unit of dislocation) / (unit of X,Y,D) / area
        U31, U32 : torch.Tensor
            Tilt. unit = (unit of dislocation) / (unit of X,Y,D) / area

    IRET : torch.Tensor (int), returned only if `return_iret` is `True`
        Return code, one value per station.
        IRET=0 means normal, IRET=1 means singular (the station coincides with
        the point source). New in the PyTorch implementation.

    Notes
    -----
    Original FORTRAN code was written by Y.Okada in Jan. 1985.
    PyTorch implementation by M.Someya in 2025.
    """

    # The only singularity of the point-source solution.  The original FORTRAN
    # has no test for it and returns NaN; DC3D0 does flag it, so the same rule
    # is applied here.
    singular = (X**2 + Y**2 + D**2) == 0.0
    IRET = torch.where(singular, 1, torch.zeros_like(X, dtype=torch.int))
    X, Y = _dummy_where(singular, X, Y)

    # Initialization
    if compute_strain:
        U1, U2, U3, U11, U12, U21, U22, U31, U32 = [torch.zeros_like(X) for _ in range(9)]
    else:
        U1, U2, U3 = [torch.zeros_like(X) for _ in range(3)]


    P = Y * CD + D * SD
    Q = Y * SD - D * CD
    X2 = X**2
    Y2 = Y**2
    XY = X * Y
    D2 = D**2
    R2 = X2 + Y2 + D2
    R = torch.sqrt(R2)
    R3 = R**3
    R5 = R**5
    QR = 3.0 * Q / R5
    RD = R + D
    R12 = 1.0 / (R * RD**2)
    R32 = R12 * (2.0 * R + D) / R2
    R33 = R12 * (3.0 * R + D) / (R2 * RD)

    A1 =  ALP * Y * (R12 - X2 * R33)
    A2 =  ALP * X * (R12 - Y2 * R33)
    A3 =  ALP * X / R3 - A2
    A4 = -ALP * XY * R32
    A5 =  ALP * (1.0 / (R * RD) - X2 * R32)


    if compute_strain:
        R4 = R**4
        S = P * SD + Q * CD
        XR  = 5.0 * X2 / R2
        YR  = 5.0 * Y2 / R2
        XYR = 5.0 * XY / R2
        DR  = 5.0 * D / R2
        R53 = R12 * (8.0 * R2 + 9.0 * R * D + 3.0 * D2) / (R4 * RD)
        R54 = R12 * (5.0 * R2 + 4.0 * R * D +       D2) / R3 * R12

        B1 =  ALP * (-3.0 * XY * R33      + 3.0 * X2 * XY * R54)
        B2 =  ALP * (1.0 / R3 - 3.0 * R12 + 3.0 * X2 * Y2 * R54)
        B3 =  ALP * (1.0 / R3 - 3.0 * X2 / R5) - B2
        B4 = -ALP * 3.0 * XY / R5 - B1
        C1 = -ALP * Y * (R32 - X2 * R53)
        C2 = -ALP * X * (R32 - Y2 * R53)
        C3 = -ALP * 3.0 * X * D / R5 - C2



    # STRIKE-SLIP CONTRIBUTION
    UN = DISL1 / PI2
    QRX = QR * X
    U1 = U1 - UN * (QRX * X + A1 * SD)
    U2 = U2 - UN * (QRX * Y + A2 * SD)
    U3 = U3 - UN * (QRX * D + A4 * SD)

    if compute_strain:
        FX = 3.0 * X / R5 * SD
        U11 = U11 - UN * ( QRX *    (2.0 - XR)          + B1 * SD)
        U12 = U12 - UN * (-QRX * XYR           + FX * X + B2 * SD)
        U21 = U21 - UN * ( QR * Y * (1.0 - XR)          + B2 * SD)
        U22 = U22 - UN * ( QRX *    (1.0 - YR) + FX * Y + B4 * SD)
        U31 = U31 - UN * ( QR * D * (1.0 - XR)          + C1 * SD)
        U32 = U32 - UN * (-QRX * DR * Y        + FX * D + C2 * SD)


    # DIP-SLIP CONTRIBUTION
    UN = DISL2 / PI2
    SDCD = SD * CD
    QRP = QR * P
    U1 = U1 - UN * (QRP * X - A3 * SDCD)
    U2 = U2 - UN * (QRP * Y - A1 * SDCD)
    U3 = U3 - UN * (QRP * D - A5 * SDCD)

    if compute_strain:
        FS = 3.0 * S / R5
        U11 = U11 - UN * ( QRP * (1.0 - XR)          - B3 * SDCD)
        U12 = U12 - UN * (-QRP * XYR        + FS * X - B1 * SDCD)
        U21 = U21 - UN * (-QRP * XYR                 - B1 * SDCD)
        U22 = U22 - UN * ( QRP * (1.0 - YR) + FS * Y - B2 * SDCD)
        U31 = U31 - UN * (-QRP * DR * X              - C3 * SDCD)
        U32 = U32 - UN * (-QRP * DR * Y     + FS * D - C1 * SDCD)


    # TENSILE-FAULT CONTRIBUTION
    UN = DISL3 / PI2
    SDSD = SD**2
    QRQ = QR * Q
    U1 = U1 + UN * (QRQ * X - A3 * SDSD)
    U2 = U2 + UN * (QRQ * Y - A1 * SDSD)
    U3 = U3 + UN * (QRQ * D - A5 * SDSD)

    if compute_strain:
        FQ = 2.0 * QR * SD
        U11 = U11 + UN * ( QRQ * (1.0 - XR)          - B3 * SDSD)
        U12 = U12 + UN * (-QRQ * XYR        + FQ * X - B1 * SDSD)
        U21 = U21 + UN * (-QRQ * XYR                 - B1 * SDSD)
        U22 = U22 + UN * ( QRQ * (1.0 - YR) + FQ * Y - B2 * SDSD)
        U31 = U31 + UN * (-QRQ * DR * X              - C3 * SDSD)
        U32 = U32 + UN * (-QRQ * DR * Y     + FQ * D - C1 * SDSD)


    if compute_strain:
        U = [U1, U2, U3, U11, U12, U21, U22, U31, U32]
    else:
        U = [U1, U2, U3]
    U = _blank_where(singular, U)

    return (U, IRET) if return_iret else U




def SRECTF(ALP, X, Y, DEP, AL, AW, SD, CD, DISL1, DISL2, DISL3,
           compute_strain=True, return_iret=False):
    """
    Surface displacements, strains and tilts due to rectangular fault in a half-space.

    Parameters
    ----------
    ALP : float or torch.Tensor
        Medium constant. myu/(lambda+myu)
    X, Y : torch.Tensor
        Coordinate of station.
    D : float or torch.Tensor
        Source depth.
    AL, AW : float or torch.Tensor
        Length and width of fault.
    SD, CD : float or torch.Tensor
        Sin, Cosine of dip-angle. 
        (CD=0.0, SD=+/-1.0 should be given for vertical fault.)
    DISL1, DISL2, DISL3 : float or torch.Tensor
        Strike-, dip- and tensile-dislocation.
    compute_strain : bool, default True
        Option to calculate the spatial derivative of the displacement. 
        New in the PyTorch implementation.
    return_iret : bool, default False
        Return the singularity flag. New in the PyTorch implementation.

    Returns
    -------
    U : list of torch.Tensor
        If `compute_strain` is `True`, 
        U is a list of 3 displacements and 6 spatial derivatives:
        [U1, U2, U3, U11, U12, U21, U22, U31, U32]
        If `False`, U is a list of 3 displacements only:
        [U1, U2, U3]
        Stations flagged by `IRET` are returned as exactly zero.

        U1, U2, U3 : torch.Tensor
            Displacement. unit = (unit of dislocation) 
        U11, U12, U21, U22 : torch.Tensor
            Strain. unit = (unit of dislocation) / (unit of X,Y, ... , AW)
        U31, U32 : torch.Tensor
            Tilt. unit = (unit of dislocation) / (unit of X,Y, ... , AW)

    IRET : torch.Tensor (int), returned only if `return_iret` is `True`
        Return code, one value per station.
        IRET=0 means normal, IRET=1 means singular (the station lies on an edge
        of the fault). New in the PyTorch implementation.

    Notes
    -----
    Original FORTRAN code was written by Y.Okada in Jan. 1985.
    PyTorch implementation by M.Someya in 2025.

    Subfunction used ... _SRECTG
    """

    # Initialization
    N_variable = 9 if compute_strain else 3
    U = [torch.zeros_like(X) for _ in range(N_variable)]


    P = Y * CD + DEP * SD
    Q = Y * SD - DEP * CD

    # Same singular-case test that DC3D uses, so the surface and the depth
    # formulation agree on which stations are unusable.
    singular = _on_fault_edge(X, X - AL, P, P - AW, Q,
                              _length_scale(AL, AW, reference=X))
    IRET = torch.where(singular, 1, torch.zeros_like(X, dtype=torch.int))


    for K in [1, 2]:
        ET = (P if (K == 1) else P - AW)
        for J in [1, 2]:
            XI = (X if (J == 1) else X - AL)
            SIGN = (1.0 if (J + K != 3) else -1.0)

            # Dummy geometry for the singular stations; see `_dummy_where`.
            XI_s, ET_s, Q_s = _dummy_where(singular, XI, ET, Q)

            DU = _SRECTG(
                ALP, XI_s, ET_s, Q_s, SD, CD, DISL1, DISL2, DISL3, compute_strain
            )

            for I in range(N_variable):
                U[I] = U[I] + SIGN * DU[I]

    U = _blank_where(singular, U)

    return (U, IRET) if return_iret else U
    

