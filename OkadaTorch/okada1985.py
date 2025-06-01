import torch

PI2 = 2.0 * torch.pi


class SPOINT:
    def __init__(self):
        pass


    def compute(self, ALP, X, Y, D, SD, CD, DISL1, DISL2, DISL3, compute_derivative):
        """
        SURFACE DISPLACEMENT,STRAIN,TILT DUE TO BURIED POINT SOURCE
        IN A SEMIINFINITE MEDIUM     CODED BY  Y.OKADA ... JAN 1985
        Converted to PyTorch by Masayoshi Someya (2025)

        INPUT
            ALP   : MEDIUM CONSTANT  MYU/(LAMBDA+MYU)
            X,Y   : COORDINATE OF STATION
            D     : SOURCE DEPTH
            SD,CD : SIN,COS OF DIP-ANGLE
                    (CD=0.D0, SD=+/-1.D0 SHOULD BE GIVEN FOR VERTICAL FAULT)
            DISL1,DISL2,DISL3 : STRIKE-, DIP- AND TENSILE-DISLOCATION
        
        OUTPUT
            U1, U2, U3      : DISPLACEMENT ( UNIT= UNIT OF DISL / AREA )
            U11,U12,U21,U22 : STRAIN       ( UNIT= UNIT OF DISL /
            U31,U32         : TILT                 UNIT OF X,Y,D /AREA )
        """

        # Initialization
        if compute_derivative:
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


        if compute_derivative:
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
        if DISL1 != 0.0:
            UN = DISL1 / PI2
            QRX = QR * X
            U1 = U1 - UN * (QRX * X + A1 * SD)
            U2 = U2 - UN * (QRX * Y + A2 * SD)
            U3 = U3 - UN * (QRX * D + A4 * SD)

            if compute_derivative:
                FX = 3.0 * X / R5 * SD
                U11 = U11 - UN * ( QRX *    (2.0 - XR)          + B1 * SD)
                U12 = U12 - UN * (-QRX * XYR           + FX * X + B2 * SD)
                U21 = U21 - UN * ( QR * Y * (1.0 - XR)          + B2 * SD)
                U22 = U22 - UN * ( QRX *    (1.0 - YR) + FX * Y + B4 * SD)
                U31 = U31 - UN * ( QR * D * (1.0 - XR)          + C1 * SD)
                U32 = U32 - UN * (-QRX * DR * Y        + FX * D + C2 * SD)


        # DIP-SLIP CONTRIBUTION
        if DISL2 != 0.0:
            UN = DISL2 / PI2
            SDCD = SD * CD
            QRP = QR * P
            U1 = U1 - UN * (QRP * X - A3 * SDCD)
            U2 = U2 - UN * (QRP * Y - A1 * SDCD)
            U3 = U3 - UN * (QRP * D - A5 * SDCD)

            if compute_derivative:
                FS = 3.0 * S / R5
                U11 = U11 - UN * ( QRP * (1.0 - XR)          - B3 * SDCD)
                U12 = U12 - UN * (-QRP * XYR        + FS * X - B1 * SDCD)
                U21 = U21 - UN * (-QRP * XYR                 - B1 * SDCD)
                U22 = U22 - UN * ( QRP * (1.0 - YR) + FS * Y - B2 * SDCD)
                U31 = U31 - UN * (-QRP * DR * X              - C3 * SDCD)
                U32 = U32 - UN * (-QRP * DR * Y     + FS * D - C1 * SDCD)


        # TENSILE-FAULT CONTRIBUTION
        if DISL3 != 0.0:
            UN = DISL3 / PI2
            SDSD = SD**2
            QRQ = QR * Q
            U1 = U1 + UN * (QRQ * X - A3 * SDSD)
            U2 = U2 + UN * (QRQ * Y - A1 * SDSD)
            U3 = U3 + UN * (QRQ * D - A5 * SDSD)

            if compute_derivative:
                FQ = 2.0 * QR * SD
                U11 = U11 + UN * ( QRQ * (1.0 - XR)          - B3 * SDSD)
                U12 = U12 + UN * (-QRQ * XYR        + FQ * X - B1 * SDSD)
                U21 = U21 + UN * (-QRQ * XYR                 - B1 * SDSD)
                U22 = U22 + UN * ( QRQ * (1.0 - YR) + FQ * Y - B2 * SDSD)
                U31 = U31 + UN * (-QRQ * DR * X              - C3 * SDSD)
                U32 = U32 + UN * (-QRQ * DR * Y     + FQ * D - C1 * SDSD)


        if compute_derivative:
            return [U1, U2, U3, U11, U12, U21, U22, U31, U32]
        else:
            return [U1, U2, U3]




class SRECTF:
    def __init__(self):
        pass

    def compute(self, ALP, X, Y, DEP, AL, AW, SD, CD, DISL1, DISL2, DISL3, compute_derivative):
        """
        SURFACE DISPLACEMENTS,STRAINS AND TILTS DUE TO RECTANGULAR
        FAULT IN A HALF-SPACE       CODED BY  Y.OKADA ... JAN 1985
        Converted to PyTorch by Masayoshi Someya (2025)

        INPUT
            ALP   : MEDIUM CONSTANT  MYU/(LAMBDA+MYU)
            X,Y   : COORDINATE OF STATION
            DEP   : SOURCE DEPTH
            AL,AW : LENGTH AND WIDTH OF FAULT
            SD,CD : SIN,COS OF DIP-ANGLE
                (CD=0.D0, SD=+/-1.D0 SHOULD BE GIVEN FOR VERTICAL FAULT)
            DISL1,DISL2,DISL3 : STRIKE-, DIP- AND TENSILE-DISLOCATION
        
        OUTPUT
            U1, U2, U3      : DISPLACEMENT ( UNIT= UNIT OF DISL     )
            U11,U12,U21,U22 : STRAIN       ( UNIT= UNIT OF DISL /
            U31,U32         : TILT                 UNIT OF X,Y,,,AW )

        SUBROUTINE USED... _SRECTG
        """

        # Initialization
        N_variable = 9 if compute_derivative else 3
        U = [torch.zeros_like(X) for _ in range(N_variable)]
        DU = [torch.zeros_like(X) for _ in range(N_variable)]


        P = Y * CD + DEP * SD
        Q = Y * SD - DEP * CD


        for K in [1, 2]:
            ET = P if (K == 1) else P - AW
            for J in [1, 2]:
                XI = X if (J == 1) else X - AL
                SIGN = 1.0 if (J + K != 3) else -1.0

                DU = self._SRECTG(
                    ALP, XI, ET, Q, SD, CD, DISL1, DISL2, DISL3, compute_derivative
                )

                for I in range(N_variable):
                    U[I] = U[I] + SIGN * DU[I]


        return U
        


    def _SRECTG(self, ALP, XI, ET, Q, SD, CD, DISL1, DISL2, DISL3, compute_derivative):
        """
        INDEFINITE INTEGRAL OF SURFACE DISPLACEMENTS, STRAINS AND TILTS
        DUE TO FINITE FAULT IN A SEMIINFINITE MEDIUM
        CODED BY  Y.OKADA ... JAN 1985
        Converted to PyTorch by Masayoshi Someya (2025)

        INPUT
            ALP     : MEDIUM CONSTANT  MYU/(LAMBDA+MYU)
            XI,ET,Q : FAULT COORDINATE
            SD,CD   : SIN,COS OF DIP-ANGLE
                    (CD=0.D0, SD=+/-1.D0 SHOULD BE GIVEN FOR VERTICAL FAULT)
            DISL1,DISL2,DISL3 : STRIKE-, DIP- AND TENSILE-DISLOCATION

        OUTPUT
            U1, U2, U3      : DISPLACEMENT ( UNIT= UNIT OF DISL    )
            U11,U12,U21,U22 : STRAIN       ( UNIT= UNIT OF DISL /
            U31,U32         : TILT                 UNIT OF XI,ET,Q )
        """

        # Initialization
        if compute_derivative:
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
        
        TT = torch.where(
            Q != 0.0, 
            torch.atan(XI * ET / (Q * R)), 
            0.0
        )
        RE = torch.where(
            RET != 0.0, 
            1.0 / RET, 
            0.0
        )
        DLE = torch.where(
            RET != 0.0, 
            torch.log(RET), 
            -torch.log(R - ET)
        )
        # Modification to prevent zero-division. 
        # RRX = 1.0 / ( R * (R + XI))
        RRX = torch.where(
            torch.abs(R * (R + XI)) < 1.0e-6,
            1.0e6, 
            1.0 / (R * (R + XI))
        )
        RRE = RE / R

        if CD != 0.0:
            # INCLINED FAULT
            TD = SD / CD
            X = torch.sqrt(XI2 + Q2)
            A5 = torch.where(
                XI == 0.0, 
                0.0, 
                ALP * 2.0 / CD * torch.atan( 
                    (ET * (X + Q * CD) + X * (R + X) * SD) / (XI * (R + X) * CD) 
                )
            )
            A4 =  ALP / CD * (torch.log(RD) - SD * DLE)
            A3 =  ALP * ( Y / RD / CD - DLE) + TD * A4
            A1 = -ALP / CD * XI / RD         - TD * A5
        else:
            # VERTICAL FAULT
            RD2 = RD**2
            A1 = -ALP / 2.0 * XI * Q / RD2
            A3 =  ALP / 2.0 * (ET / RD + Y * Q / RD2 - DLE)
            A4 = -ALP * Q / RD
            A5 = -ALP * XI * SD / RD

        A2 = -ALP * DLE - A3


        if compute_derivative:
            RRD = 1.0 / (R * RD)
            AXI = (2.0 * R + XI) * RRX**2 / R
            AET = (2.0 * R + ET) * RRE**2 / R
            R3 = R**3

            if CD != 0.0:
                # INCLINED FAULT
                C1 = ALP / CD * XI * (RRD - SD * RRE)
                C3 = ALP / CD * (Q * RRE - Y * RRD)
                B1 = ALP / CD * (XI2 * RRD - 1.0) / RD - TD * C3
                B2 = ALP / CD * XI * Y * RRD / RD      - TD * C1
            else:
                # VERTICAL FAULT
                B1 = ALP / 2.0 * Q / RD2 * (2.0 * XI2 * RRD - 1.0)
                B2 = ALP / 2.0 * XI * SD / RD2 * (2.0 * Q2 * RRD - 1.0)
                C1 = ALP * XI * Q * RRD / RD
                C3 = ALP * SD / RD * (XI2 * RRD - 1.0)

            B3 = -ALP * XI * RRE - B2
            B4 = -ALP * (CD / R + Q * SD * RRE) - B1
            C2 = ALP * (-SD / R + Q * CD * RRE) - C3




        # STRIKE-SLIP CONTRIBUTION
        if DISL1 != 0.0:
            UN = DISL1 / PI2
            REQ = RRE * Q
            U1 = U1 - UN * (REQ * XI + TT          + A1 * SD)
            U2 = U2 - UN * (REQ * Y  + Q * CD * RE + A2 * SD)
            U3 = U3 - UN * (REQ * D  + Q * SD * RE + A4 * SD)

            if compute_derivative:
                U11 = U11 + UN * (XI2 * Q * AET - B1 * SD)
                U12 = U12 + UN * (XI2 * XI * (D / (ET2 + Q2) / R3 - AET * SD) - B2 * SD)
                U21 = U21 + UN * (XI * Q / R3 * CD + (XI * Q2 * AET - B2) * SD)
                U22 = U22 + UN * (Y * Q / R3 * CD + (Q * SD * (Q2 * AET - 2.0 * RRE) - (XI2 + ET2) / R3 * CD - B4) * SD)
                U31 = U31 + UN * (-XI * Q2 * AET * CD + (XI * Q / R3 - C1) * SD)
                U32 = U32 + UN * (D * Q / R3 * CD + (XI2 * Q * AET * CD - SD / R + Y * Q / R3 - C2) * SD)


        # DIP-SLIP CONTRIBUTION
        if DISL2 != 0.0:
            UN = DISL2 / PI2
            SDCD = SD * CD
            U1 = U1 - UN * (Q / R                 - A3 * SDCD)
            U2 = U2 - UN * (Y * Q * RRX + CD * TT - A1 * SDCD)
            U3 = U3 - UN * (D * Q * RRX + SD * TT - A5 * SDCD)

            if compute_derivative:
                U11 = U11 + UN * (XI * Q / R3                + B3 * SDCD)
                U12 = U12 + UN * (Y * Q / R3  - SD / R       + B1 * SDCD)
                U21 = U21 + UN * (Y * Q / R3  + Q * CD * RRE + B1 * SDCD)
                U22 = U22 + UN * (Y * Y * Q * AXI - (2.0 * Y * RRX + XI * CD * RRE) * SD + B2 * SDCD)
                U31 = U31 + UN * (D * Q / R3  + Q * SD * RRE + C3 * SDCD)
                U32 = U32 + UN * (Y * D * Q * AXI - (2.0 * D * RRX + XI * SD * RRE) * SD + C1 * SDCD)


        # TENSILE-FAULT CONTRIBUTION
        if DISL3 != 0.0:
            UN = DISL3 / PI2
            SDSD = SD**2
            U1 = U1 + UN * (Q2 * RRE                                - A3 * SDSD)
            U2 = U2 + UN * (-D * Q * RRX - SD * (XI * Q * RRE - TT) - A1 * SDSD)
            U3 = U3 + UN * ( Y * Q * RRX + CD * (XI * Q * RRE - TT) - A5 * SDSD)

            if compute_derivative:
                U11 = U11 - UN * (XI * Q2 * AET                    + B3 * SDSD)
                U12 = U12 - UN * (-D * Q / R3 - XI2 * Q * AET * SD + B1 * SDSD)
                U21 = U21 - UN * (Q2 * (CD / R3 + Q * AET * SD)    + B1 * SDSD)
                U22 = U22 - UN * ((Y * CD - D * SD) * Q2 * AXI - 2.0 * Q * SD * CD * RRX - (XI * Q2 * AET - B2) * SDSD)
                U31 = U31 - UN * (Q2 * (SD / R3 - Q * AET * CD) + C3 * SDSD)
                U32 = U32 - UN * ((Y * SD + D * CD) * Q2 * AXI + XI * Q2 * AET * SD * CD - (2.0 * Q * RRX - C1) * SDSD)



        if compute_derivative:
            return [U1, U2, U3, U11, U12, U21, U22, U31, U32]
        else:
            return [U1, U2, U3]
