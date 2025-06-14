import torch
from .utils import _UA0, _UB0, _UC0, _UA, _UB, _UC, COMMON0, COMMON1, COMMON2

PI2 = 2.0 * torch.pi
EPS = 1.0e-6




def DC3D0(ALPHA, X, Y, Z, DEPTH, DIP, POT1, POT2, POT3, POT4, 
          compute_strain=True, is_degree=True):
    """
    Displacement and strain at depth due to buried point source 
    in a semiinfinite medium.

    Parameters
    ----------
    ALPHA : float or torch.Tensor
        Medium constant. (lambda+myu)/(lambda+2*myu)
    X, Y, Z : torch.Tensor
        Coordinate of observing point.
    DEPTH : float or torch.Tensor
        Source depth.
    DIP : torch.Tensor
        Dip-angle.
    POT1, POT2, POT3, POT4 : float or torch.Tensor
        Strike-, dip-, tensile- and inflate-potency.
        potency = (moment of double-couple)/myu for POT1,2
        potency = (intensity of isotropic part)/lambda for POT3
        potency = (intensity of linear dipole)/myu for POT4
    compute_strain : bool, dafault True
        Option to calculate the spatial derivative of the displacement. 
        New in the PyTorch implementation.
    is_degree : bool, dafault True
        Flag if `DIP` is in degree or not (= in radian). 
        New in the PyTorch implementation.

    Returns
    -------
    U : list of torch.Tensor
        If `compute_strain` is `True`, 
        U is a list of 3 displacements and 9 spatial derivatives:
        [UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]
        If `False`, U is a list of 3 displacements only:
        [UX, UY, UZ]

        UX, UY, UZ : torch.Tensor
            Displacement. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**2
        UXX, UYX, UZX : torch.Tensor
            X-derivative. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**3
        UXY, UYY, UZY : torch.Tensor
            Y-derivative. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**3
        UXZ, UYZ, UZZ : torch.Tensor
            Z-derivative. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**3

    IRET : torch.Tensor (int)
        Return code.
        IRET=0 means normal,
        IRET=1 means singular,
        IRET=2 means positive z was given.


    Notes
    -----
    Original FORTRAN code was written by Y.Okada in Sep.1991, 
    revised in Nov.1991, May.2002.
    PyTorch implementation by M.Someya, 2025.
    """
    

    # Initialization
    N_variable = 12 if compute_strain else 3
    U = [torch.zeros_like(X) for _ in range(N_variable)]
    DUA = [torch.zeros_like(X) for _ in range(N_variable)]
    DUB = [torch.zeros_like(X) for _ in range(N_variable)]
    DUC = [torch.zeros_like(X) for _ in range(N_variable)]
    IRET = torch.zeros_like(X, dtype=torch.int)

    IRET = torch.where(
        Z > 0.0, 
        2,
        IRET
    )
    
    C0 = COMMON0()
    C0.DCCON0(ALPHA, DIP, is_degree)


    # REAL-SOURCE CONTRIBUTION
    DD = DEPTH + Z
    C1 = COMMON1()
    C1.DCCON1(X, Y, DD, C0)


    # IN CASE OF SINGULAR (R=0)
    IRET = torch.where(
        C1.R==0.0, 
        1,
        IRET
    )
    
    DUA = _UA0(X, Y, DD, POT1, POT2, POT3, POT4, C0, C1, compute_strain)
    if compute_strain:
        for I in range(9):
            U[I] = U[I] - DUA[I]
        for I in range(3):
            U[I+9] = U[I+9] + DUA[I+9]
    else:
        for I in range(3):
            U[I] = U[I] - DUA[I]


    # IMAGE-SOURCE CONTRIBUTION
    DD = DEPTH - Z
    C1.DCCON1(X, Y, DD, C0)

    DUA = _UA0(X, Y, DD, POT1, POT2, POT3, POT4, C0, C1, compute_strain)
    DUB = _UB0(X, Y, DD, Z, POT1, POT2, POT3, POT4, C0, C1, compute_strain)
    DUC = _UC0(X, Y, DD, Z, POT1, POT2, POT3, POT4, C0, C1, compute_strain)

    for I in range (N_variable):
        DU = DUA[I] + DUB[I] + Z * DUC[I]
        if I >= 9:
            DU = DU + DUC[I-9]
        U[I] = U[I] + DU

    return U, IRET
    




def DC3D(ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3, 
         compute_strain=True, is_degree=True):
    """
    Displacement and strain at depth due to buried finite fault 
    in a semiinfinite medium.

    Parameters
    ----------
    ALPHA : float or torch.Tensor
        Medium constant. (lambda+myu)/(lambda+2*myu)
    X, Y, Z : torch.Tensor
        Coordinate of observing point.
    DEPTH : float or torch.Tensor
        Depth of reference point.
    DIP : torch.Tensor
        Dip-angle.
    AL1, AL2 : float or torch.Tensor
        Fault length range.
    AW1, AW2 : float or torch.Tensor
        Fault width range.
    DISL1, DISL2, DISL3 : float or torch.Tensor
        Strike-, dip-, tensile-dislocations.
    compute_strain : bool, dafault True
        Option to calculate the spatial derivative of the displacement.
        New in the PyTorch implementation.
    is_degree : bool, dafault True
        Flag if `DIP` is in degree or not (= in radian). 
        New in the PyTorch implementation.

    Returns
    -------
    U : list of torch.Tensor
        If `compute_strain` is `True`, 
        U is a list of 3 displacements and 9 spatial derivatives:
        [UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]
        If `False`, U is a list of 3 displacements only:
        [UX, UY, UZ]

        UX, UY, UZ : torch.Tensor
            Displacement. unit = (unit of dislocation)
        UXX, UYX, UZX : torch.Tensor
            X-derivative. unit = (dislocation) / (unit of X,Y,Z,DEPTH,AL,AW)
        UXY, UYY, UZY : torch.Tensor
            Y-derivative. unit = (dislocation) / (unit of X,Y,Z,DEPTH,AL,AW)
        UXZ, UYZ, UZZ : torch.Tensor
            Z-derivative. unit = (dislocation) / (unit of X,Y,Z,DEPTH,AL,AW)
            
    IRET : torch.Tensor (whose dtype is torch.int)
        Return code.
        IRET=0 means normal,
        IRET=1 means singular,
        IRET=2 means positive z was given.


    Notes
    -----
    Original FORTRAN code was written by Y.Okada in Sep.1991, 
    revised in Nov.1991, Apr.1992, May.1993, Jul.1993, May.2002.
    PyTorch implementation by M.Someya, 2025.
    """


    # Initialization
    N_variable = 12 if compute_strain else 3
    U = [torch.zeros_like(X) for _ in range(N_variable)]
    DU = [torch.zeros_like(X) for _ in range(N_variable)]
    DUA = [torch.zeros_like(X) for _ in range(N_variable)]
    DUB = [torch.zeros_like(X) for _ in range(N_variable)]
    DUC = [torch.zeros_like(X) for _ in range(N_variable)]
    XI = [torch.zeros_like(X) for _ in range(2)]
    ET = [torch.zeros_like(X) for _ in range(2)]
    KXI = [torch.zeros_like(X, dtype=torch.int) for _ in range(2)]
    KET = [torch.zeros_like(X, dtype=torch.int) for _ in range(2)]
    IRET = torch.zeros_like(X, dtype=torch.int)

    IRET = torch.where(
        Z > 0.0, 
        2,
        IRET
    )

    C0 = COMMON0()
    C0.DCCON0(ALPHA, DIP, is_degree)
    SD, CD = C0.SD, C0.CD 


    XI[0] = torch.where(
        torch.abs(X - AL1) < EPS,
        0.0,
        X - AL1
    )
    XI[1] = torch.where(
        torch.abs(X - AL2) < EPS,
        0.0,
        X - AL2
    )

    
    # REAL-SOURCE CONTRIBUTION
    D = DEPTH + Z
    P = Y * CD + D * SD
    Q = torch.where(
        torch.abs(Y * SD - D * CD) < EPS,
        0.0,
        Y * SD - D * CD
    )
    ET[0] = torch.where(
        torch.abs(P - AW1) < EPS,
        0.0,
        P - AW1
    )
    ET[1] = torch.where(
        torch.abs(P - AW2) < EPS,
        0.0,
        P - AW2
    )


    # REJECT SINGULAR CASE
    # ON FAULT EDGE
    mask1 = torch.logical_and(Q == 0.0, torch.logical_or( 
        torch.logical_and(XI[0] * XI[1] <= 0.0, ET[0] * ET[1] == 0.0),
        torch.logical_and(ET[0] * ET[1] <= 0.0, XI[0] * XI[1] == 0.0)
    ))
    IRET = torch.where(
        mask1, 
        1,
        IRET
    )

    
    ## ON NEGATIVE EXTENSION OF FAULT EDGE
    R12 = torch.sqrt(XI[0]**2 + ET[1]**2 + Q**2)
    R21 = torch.sqrt(XI[1]**2 + ET[0]**2 + Q**2)
    R22 = torch.sqrt(XI[1]**2 + ET[1]**2 + Q**2)

    KXI[0] = torch.where(
        torch.logical_and(XI[0] < 0.0, R21 + XI[1] < EPS),
        1,
        0
    )
    KXI[1] = torch.where(
        torch.logical_and(XI[0] < 0.0, R22 + XI[1] < EPS),
        1,
        0
    )
    KET[0] = torch.where(
        torch.logical_and(ET[0] < 0.0, R12 + ET[1] < EPS),
        1,
        0
    )
    KET[1] = torch.where(
        torch.logical_and(ET[0] < 0.0, R22 + ET[1] < EPS),
        1,
        0
    )
    
    C2 = COMMON2()

    for K in range(2):
        for J in range(2):
            C2.DCCON2(XI[J], ET[K], Q, SD, CD, KXI[K], KET[J])
            DUA = _UA(XI[J], ET[K], Q, DISL1, DISL2, DISL3, C0, C2, compute_strain)

            if compute_strain:
                for I in range(0, 10, 3):
                    DU[I]   = -DUA[I]
                    DU[I+1] = -DUA[I+1] * CD + DUA[I+2] * SD
                    DU[I+2] = -DUA[I+1] * SD - DUA[I+2] * CD
                    if I >= 9:
                        DU[I]   = -DU[I]
                        DU[I+1] = -DU[I+1]
                        DU[I+2] = -DU[I+2]
            else:
                DU[0] = -DUA[0]
                DU[1] = -DUA[1] * CD + DUA[2] * SD
                DU[2] = -DUA[1] * SD - DUA[2] * CD


            for I in range(N_variable):
                if (J + K == 1):
                    U[I] = U[I] - DU[I]
                else:
                    U[I] = U[I] + DU[I]



    # IMAGE-SOURCE CONTRIBUTION
    D = DEPTH - Z
    P = Y * CD + D * SD
    Q = torch.where(
        torch.abs(Y * SD - D * CD) < EPS,
        0.0,
        Y * SD - D * CD
    )
    ET[0] = torch.where(
        torch.abs(P - AW1) < EPS,
        0.0,
        P - AW1
    )
    ET[1] = torch.where(
        torch.abs(P - AW2) < EPS,
        0.0,
        P - AW2
    )


    # REJECT SINGULAR CASE
    # ON FAULT EDGE
    mask2 = torch.logical_and(Q == 0.0, torch.logical_or( 
        torch.logical_and(XI[0] * XI[1] <= 0.0, ET[0] * ET[1] == 0.0),
        torch.logical_and(ET[0] * ET[1] <= 0.0, XI[0] * XI[1] == 0.0)
    ))
    IRET = torch.where(
        mask2, 
        1,
        IRET
    )
    
    
    ## ON NEGATIVE EXTENSION OF FAULT EDGE
    R12 = torch.sqrt(XI[0]**2 + ET[1]**2 + Q**2)
    R21 = torch.sqrt(XI[1]**2 + ET[0]**2 + Q**2)
    R22 = torch.sqrt(XI[1]**2 + ET[1]**2 + Q**2)
    
    KXI[0] = torch.where(
        torch.logical_and(XI[0] < 0.0, R21 + XI[1] < EPS),
        1,
        0
    )
    KXI[1] = torch.where(
        torch.logical_and(XI[0] < 0.0, R22 + XI[1] < EPS),
        1,
        0
    )
    KET[0] = torch.where(
        torch.logical_and(ET[0] < 0.0, R12 + ET[1] < EPS),
        1,
        0
    )
    KET[1] = torch.where(
        torch.logical_and(ET[0] < 0.0, R22 + ET[1] < EPS),
        1,
        0
    )
    


    for K in range(2):
        for J in range(2):
            C2.DCCON2(XI[J], ET[K], Q, SD, CD, KXI[K], KET[J])
            DUA = _UA(XI[J], ET[K], Q, DISL1, DISL2, DISL3, C0, C2, compute_strain)
            DUB = _UB(XI[J], ET[K], Q, DISL1, DISL2, DISL3, C0, C2, compute_strain)
            DUC = _UC(XI[J], ET[K], Q, Z, DISL1, DISL2, DISL3, C0, C2, compute_strain)

            if compute_strain:
                for I in range(0, 10, 3):
                    DU[I]   = DUA[I] + DUB[I] + Z * DUC[I]
                    DU[I+1] = (DUA[I+1] + DUB[I+1] + Z * DUC[I+1]) * CD - (DUA[I+2] + DUB[I+2] + Z * DUC[I+2]) * SD
                    DU[I+2] = (DUA[I+1] + DUB[I+1] - Z * DUC[I+1]) * SD + (DUA[I+2] + DUB[I+2] - Z * DUC[I+2]) * CD
                    if I >= 9:
                        DU[ 9] = DU[ 9] + DUC[0]
                        DU[10] = DU[10] + DUC[1] * CD - DUC[2] * SD
                        DU[11] = DU[11] - DUC[1] * SD - DUC[2] * CD
            else:
                DU[0] =  DUA[0] + DUB[0] + Z * DUC[0]
                DU[1] = (DUA[1] + DUB[1] + Z * DUC[1]) * CD - (DUA[2] + DUB[2] + Z * DUC[2]) * SD
                DU[2] = (DUA[1] + DUB[1] - Z * DUC[1]) * SD + (DUA[2] + DUB[2] - Z * DUC[2]) * CD


            for I in range(N_variable):
                if (J + K == 1):
                    U[I] = U[I] - DU[I]
                else:
                    U[I] = U[I] + DU[I]
                    


    return U, IRET
