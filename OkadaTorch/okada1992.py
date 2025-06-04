import torch
from .utils import _UA0, _UB0, _UC0, _UA, _UB, _UC, COMMON0, COMMON1, COMMON2

PI2 = 2.0 * torch.pi
EPS = 1e-6




def DC3D0(ALPHA, X, Y, Z, DEPTH, DIP, POT1, POT2, POT3, POT4, compute_strain):
    """
    DISPLACEMENT AND STRAIN AT DEPTH
    DUE TO BURIED POINT SOURCE IN A SEMIINFINITE MEDIUM
        CODED BY  Y.OKADA ... SEP.1991
        REVISED     NOV.1991, MAY.2002
        Converted to PyTorch by Masayoshi Someya (2025)

    INPUT
        ALPHA : MEDIUM CONSTANT  (LAMBDA+MYU)/(LAMBDA+2*MYU)
        X,Y,Z : COORDINATE OF OBSERVING POINT
        DEPTH : SOURCE DEPTH
        DIP   : DIP-ANGLE (DEGREE)
        POT1-POT4 : STRIKE-, DIP-, TENSILE- AND INFLATE-POTENCY
            POTENCY=(  MOMENT OF DOUBLE-COUPLE  )/MYU     FOR POT1,2
            POTENCY=(INTENSITY OF ISOTROPIC PART)/LAMBDA  FOR POT3
            POTENCY=(INTENSITY OF LINEAR DIPOLE )/MYU     FOR POT4

    OUTPUT
        UX, UY, UZ  : DISPLACEMENT ( UNIT=(UNIT OF POTENCY) /
                    :                     (UNIT OF X,Y,Z,DEPTH)**2  )
        UXX,UYX,UZX : X-DERIVATIVE ( UNIT= UNIT OF POTENCY) /
        UXY,UYY,UZY : Y-DERIVATIVE        (UNIT OF X,Y,Z,DEPTH)**3  )
        UXZ,UYZ,UZZ : Z-DERIVATIVE
    """

    if (Z > 0.0).any():
        raise ValueError("POSITIVE Z WAS GIVEN (Z>0; IRET=2)")
    

    # Initialization
    N_variable = 12 if compute_strain else 3
    U = [torch.zeros_like(X) for _ in range(N_variable)]
    DUA = [torch.zeros_like(X) for _ in range(N_variable)]
    DUB = [torch.zeros_like(X) for _ in range(N_variable)]
    DUC = [torch.zeros_like(X) for _ in range(N_variable)]
    

    C0 = COMMON0()
    C0.DCCON0(ALPHA, DIP)


    # REAL-SOURCE CONTRIBUTION
    DD = DEPTH + Z
    C1 = COMMON1()
    C1.DCCON1(X, Y, DD, C0)


    # IN CASE OF SINGULAR (R=0)
    if (C1.R==0.0).any():
        raise ValueError("SINGULAR (R=0; IRET=1)")
    
    DUA = _UA0(X, Y, DD, POT1, POT2, POT3, POT4, C0, C1, compute_strain)
    for I in range(9):
        U[I] = U[I] - DUA[I]
    for I in range(3):
        U[I+9] = U[I+9] + DUA[I+9]



    # IMAGE-SOURCE CONTRIBUTION
    DD = DEPTH - Z
    C1.DCCON1(X, Y, DD, C0)

    DUA = _UA0(X, Y, DD, POT1, POT2, POT3, POT4, C0, C1, compute_strain)
    DUB = _UB0(X, Y, DD, Z, POT1, POT2, POT3, POT4, C0, C1, compute_strain)
    DUC = _UC0(X, Y, DD, Z, POT1, POT2, POT3, POT4, C0, C1, compute_strain)

    for I in range (N_variable):
        DU = DUA[I] + DUB[I] + Z * DUC[I]
        if I >= 10:
            DU = DU + DUC[I-9]
        U[I] = U[I] + DU

    return U
    




def DC3D(ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3, compute_strain):
    """
    DISPLACEMENT AND STRAIN AT DEPTH
    DUE TO BURIED FINITE FAULT IN A SEMIINFINITE MEDIUM
        CODED BY  Y.OKADA ... SEP.1991
        REVISED ... NOV.1991, APR.1992, MAY.1993,JUL.1993, MAY.2002
        Converted to PyTorch by Masayoshi Someya (2025)

    INPUT
        ALPHA : MEDIUM CONSTANT  (LAMBDA+MYU)/(LAMBDA+2*MYU)
        X,Y,Z : COORDINATE OF OBSERVING POINT
        DEPTH : DEPTH OF REFERENCE POINT
        DIP   : DIP-ANGLE (DEGREE)
        AL1,AL2   : FAULT LENGTH RANGE
        AW1,AW2   : FAULT WIDTH RANGE
        DISL1-DISL3 : STRIKE-, DIP-, TENSILE-DISLOCATIONS

    OUTPUT
        UX, UY, UZ  : DISPLACEMENT ( UNIT=(UNIT OF DISL)
        UXX,UYX,UZX : X-DERIVATIVE ( UNIT=(UNIT OF DISL) /
        UXY,UYY,UZY : Y-DERIVATIVE        (UNIT OF X,Y,Z,DEPTH,AL,AW) )
        UXZ,UYZ,UZZ : Z-DERIVATIVE
        IRET        : RETURN CODE
                    :   =0....NORMAL
                    :   =1....SINGULAR
                    :   =2....POSITIVE Z WAS GIVEN
    """
    
    if (Z > 0.0).any():
        raise ValueError("POSITIVE Z WAS GIVEN (Z>0; IRET=2)")


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


    C0 = COMMON0()
    C0.DCCON0(ALPHA, DIP)
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
    ## ON FAULT EDGE
    if (Q == 0.0 and ( 
        (XI[0] * XI[1] <= 0.0 and ET[0] * ET[1] == 0.0) or 
        (ET[0] * ET[1] <= 0.0 and XI[0] * XI[1] == 0.0)
        )).any():
        raise ValueError("SINGULAR (IRET=1)")


    
    ## ON NEGATIVE EXTENSION OF FAULT EDGE
    R12 = torch.sqrt(XI[0]**2 + ET[1]**2 + Q**2)
    R21 = torch.sqrt(XI[1]**2 + ET[0]**2 + Q**2)
    R22 = torch.sqrt(XI[1]**2 + ET[1]**2 + Q**2)
    KXI[0] = torch.where(
        XI[0] < 0.0 and R21 + XI[1] < EPS,
        1,
        0
    )
    KXI[1] = torch.where(
        XI[0] < 0.0 and R22 + XI[1] < EPS,
        1,
        0
    )
    KET[0] = torch.where(
        ET[0] < 0.0 and R12 + ET[1] < EPS,
        1,
        0
    )
    KET[1] = torch.where(
        ET[0] < 0.0 and R22 + ET[1] < EPS,
        1,
        0
    )
    
    C2 = COMMON2()

    for K in range(2):
        for J in range(2):
            C2.DCCON2(XI[J], ET[K], Q, SD, CD, KXI[K], KET[J])
            DUA = _UA(XI[J], ET[K], Q, DISL1, DISL2, DISL3, C0, C2)

            for I in range(0, 10, 3):
                DU[I]   = -DUA[I]
                DU[I+1] = -DUA[I+1] * CD + DUA[I+2] * SD
                DU[I+2] = -DUA[I+1] * SD - DUA[I+2] * CD
                if not I < 10:
                    DU[I]   = -DU[I]
                    DU[I+1] = -DU[I+1]
                    DU[I+2] = -DU[I+2]

            for I in range(12):
                if (J + K != 1):
                    U[I] = U[I] + DU[I]
                else:
                    U[I] = U[I] - DU[I]



    # IMAGE-SOURCE CONTRIBUTION
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
    ## ON FAULT EDGE
    if (Q == 0.0 and ( 
        (XI[0] * XI[1] <= 0.0 and ET[0] * ET[1] == 0.0) or 
        (ET[0] * ET[1] <= 0.0 and XI[0] * XI[1] == 0.0)
        )).any():
        raise ValueError("SINGULAR (IRET=1)")


    
    
    ## ON NEGATIVE EXTENSION OF FAULT EDGE
    R12 = torch.sqrt(XI[0]**2 + ET[1]**2 + Q**2)
    R21 = torch.sqrt(XI[1]**2 + ET[0]**2 + Q**2)
    R22 = torch.sqrt(XI[1]**2 + ET[1]**2 + Q**2)
    KXI[0] = torch.where(
        XI[0] < 0.0 and R21 + XI[1] < EPS,
        1,
        0
    )
    KXI[1] = torch.where(
        XI[0] < 0.0 and R22 + XI[1] < EPS,
        1,
        0
    )
    KET[0] = torch.where(
        ET[0] < 0.0 and R12 + ET[1] < EPS,
        1,
        0
    )
    KET[1] = torch.where(
        ET[0] < 0.0 and R22 + ET[1] < EPS,
        1,
        0
    )
    


    for K in range(2):
        for J in range(2):
            C2.DCCON2(XI[J], ET[K], Q, SD, CD, KXI[K], KET[J])
            DUA = _UA(XI[J], ET[K], Q, DISL1, DISL2, DISL3, C0, C2)
            DUB = _UB(XI[J], ET[K], Q, DISL1, DISL2, DISL3, C0, C2)
            DUC = _UC(XI[J], ET[K] ,Q, Z, DISL1, DISL2, DISL3, C0, C2)

            for I in range(0, 10, 3):
                DU[I]   = DUA[I] + DUB[I] + Z * DUC[I]
                DU[I+1] = (DUA[I+1] + DUB[I+1] + Z * DUC[I+1]) * CD - (DUA[I+2] + DUB[I+2] + Z * DUC[I+2]) * SD
                DU[I+2] = (DUA[I+1] + DUB[I+1] - Z * DUC[I+1]) * SD + (DUA[I+2] + DUB[I+2] - Z * DUC[I+2]) * CD
                if not I < 10:
                    DU[10] = DU[10] + DUC[1]
                    DU[11] = DU[11] + DUC[2] * CD - DUC[3] * SD
                    DU[12] = DU[12] - DUC[2] * SD - DUC[3] * CD

            for I in range(12):
                if (J + K != 1):
                    U[I] = U[I] + DU[I]
                else:
                    U[I] = U[I] - DU[I]
                    


    return U

