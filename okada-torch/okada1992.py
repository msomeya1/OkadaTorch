import torch

PI2 = 2.0 * torch.pi
EPS = 1e-6

class COMMON0:
    def __init__(self):
        self.ALP1 = None
        self.ALP2 = None
        self.ALP3 = None
        self.ALP4 = None
        self.ALP5 = None
        self.SD = None
        self.CD = None
        self.SDSD = None
        self.CDCD = None
        self.SDCD = None
        self.S2D = None
        self.C2D = None


    def DCCON0(self, ALPHA, DIP):
        """
        CALCULATE MEDIUM CONSTANTS AND FAULT-DIP CONSTANTS

        INPUT
        ALPHA : MEDIUM CONSTANT  (LAMBDA+MYU)/(LAMBDA+2*MYU)
        DIP   : DIP-ANGLE (DEGREE)
        ### CAUTION ### 
        IF COS(DIP) IS SUFFICIENTLY SMALL, IT IS SET TO ZERO
        """

        self.ALP1 = (1.0 - ALPHA) / 2.0
        self.ALP2 = ALPHA / 2.0
        self.ALP3 = (1.0 - ALPHA) / ALPHA
        self.ALP4 = 1.0 - ALPHA
        self.ALP5 = ALPHA

        SD = torch.sin(torch.deg2rad(DIP))
        CD = torch.cos(torch.deg2rad(DIP))

        mask = (torch.abs(CD) < EPS)
        self.SD = torch.where(mask, torch.sign(SD), SD)
        self.CD = torch.where(mask, torch.tensor(0.0), CD)

        self.SDSD = self.SD**2
        self.CDCD = self.CD**2
        self.SDCD = self.SD * self.CD
        self.S2D = 2.0 * self.SDCD
        self.C2D = self.CDCD - self.SDSD



class COMMON1:
    def __init__(self):
        self.P = None
        self.Q = None
        self.S = None
        self.T = None
        self.XY = None
        self.X2 = None
        self.Y2 = None
        self.D2 = None
        self.R = None
        self.R2 = None
        self.R3 = None
        self.R5 = None
        self.QR = None
        self.QRX = None
        self.A3 = None
        self.A5 = None
        self.B3 = None
        self.C3 = None
        self.UY = None
        self.VY = None
        self.WY = None
        self.UZ = None
        self.VZ = None
        self.WZ = None

    
    def DCCON1(self, X, Y, D, C0):
        """
        CALCULATE STATION GEOMETRY CONSTANTS FOR POINT SOURCE
        
        INPUT
        X,Y,D : STATION COORDINATES IN FAULT SYSTEM
        ### CAUTION ### 
        IF X,Y,D ARE SUFFICIENTLY SMALL, THEY ARE SET TO ZERO
        """

        SD, CD = C0.SD, C0.CD 

        X = torch.where(
            torch.abs(X) < EPS,
            0.0,
            X
        )
        Y = torch.where(
            torch.abs(Y) < EPS,
            0.0,
            Y
        )
        D = torch.where(
            torch.abs(D) < EPS,
            0.0,
            D
        )

        self.P = Y * CD + D * SD
        self.Q = Y * SD - D * CD
        self.S = self.P * SD + self.Q * CD
        self.T = self.P * CD - self.Q * SD
        self.XY = X * Y
        self.X2 = X**2
        self.Y2 = Y**2
        self.D2 = D**2
        self.R2 = self.X2 + self.Y2 + self.D2
        self.R = torch.sqrt(self.R2)
        
        if (self.R == 0.0):
            return 
        
        self.R3 = self.R**3
        self.R5 = self.R**5

        self.A3 = 1.0 - 3.0 * self.X2 / self.R2
        self.A5 = 1.0 - 5.0 * self.X2 / self.R2
        self.B3 = 1.0 - 3.0 * self.Y2 / self.R2
        self.C3 = 1.0 - 3.0 * self.D2 / self.R2

        self.QR = 3.0 * self.Q / self.R5
        self.QRX = 5.0 * self.QR * X / self.R2

        self.UY = SD - 5.0 * Y * self.Q / self.R2
        self.UZ = CD + 5.0 * D * self.Q / self.R2
        self.VY = self.S - 5.0 * Y * self.P * self.Q / self.R2
        self.VZ = self.T + 5.0 * D * self.P * self.Q / self.R2
        self.WY = self.UY + SD
        self.WZ = self.UZ + CD




class COMMON2:
    def __init__(self):
        self.XI2 = None
        self.ET2 = None
        self.Q2 = None
        self.R = None
        self.R2 = None
        self.R3 = None
        self.R5 = None
        self.Y = None
        self.D = None
        self.TT = None
        self.ALX = None
        self.ALE = None
        self.X11 = None
        self.Y11 = None
        self.X32 = None
        self.Y32 = None
        self.EY = None
        self.EZ = None
        self.FY = None
        self.FZ = None
        self.GY = None
        self.GZ = None
        self.HY = None
        self.HZ = None


    def DCCON2(self, XI, ET, Q, SD, CD, KXI, KET):
        """
        CALCULATE STATION GEOMETRY CONSTANTS FOR FINITE SOURCE

        INPUT
            XI,ET,Q : STATION COORDINATES IN FAULT SYSTEM
            SD,CD   : SIN, COS OF DIP-ANGLE
            KXI,KET : KXI=1, KET=1 MEANS R+XI<EPS, R+ET<EPS, RESPECTIVELY

        ### CAUTION ### 
        IF XI,ET,Q ARE SUFFICIENTLY SMALL, THEY ARE SET TO ZER0
        """


        if (torch.abs(XI) < EPS):
            XI=0.0
        if (torch.abs(ET) < EPS):
            ET=0.0
        if (torch.abs(Q) < EPS):
            Q=0.0
        

        self.XI2 = XI**2
        self.ET2 = ET**2
        self.Q2 = Q**2
        self.R2 = self.XI2 + self.ET2 + self.Q2
        self.R = torch.sqrt(self.R2)
        if (self.R == 0.0):
            return 
        
        self.R3 = self.R**3
        self.R5 = self.R**5
        self.Y = ET * CD + Q * SD
        self.D = ET * SD - Q * CD

        self.TT = torch.where(
            Q != 0.0, 
            torch.atan(XI * ET / (Q * self.R)), 
            0.0
        )


        if (KXI == 1):
            self.ALX = -torch.log(self.R - XI)
            self.X11 = 0.0
            self.X32 = 0.0
        else:
            RXI = self.R + XI
            self.ALX = torch.log(RXI)
            self.X11 = 1.0 / (self.R * RXI)
            self.X32 = (self.R + RXI) * self.X11**2 / self.R

        if (KET == 1):
            self.ALE = -torch.log(self.R - ET)
            self.Y11 = 0.0
            self.Y32 = 0.0
        else:
            RET = self.R + ET
            self.ALE = torch.log(RET)
            self.Y11 = 1.0 / (self.R * RET)
            self.Y32 = (self.R + RET) * self.Y11**2 / self.R


        self.EY = SD / self.R - self.Y * Q / self.R3
        self.EZ = CD / self.R + self.D * Q / self.R3
        self.FY = self.D / self.R3 + self.XI2 * self.Y32 * SD
        self.FZ = self.Y / self.R3 + self.XI2 * self.Y32 * CD
        self.GY = 2.0 * self.X11 * SD - self.Y * Q * self.X32
        self.GZ = 2.0 * self.X11 * CD + self.D * Q * self.X32
        self.HY = self.D * Q * self.X32 + XI * Q * self.Y32 * SD
        self.HZ = self.Y * Q * self.X32 + XI * Q * self.Y32 * CD




class DC3D0:
    def __init__(self):
        pass


    def compute(self, ALPHA, X, Y, Z, DEPTH, DIP, POT1, POT2, POT3, POT4, compute_derivative):
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

        if (Z > 0.0):
            raise ValueError("POSITIVE Z WAS GIVEN (Z>0; IRET=2)")
        

        # Initialization
        N_variable = 12 if compute_derivative else 3
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
        if (C1.R==0.0):
            raise ValueError("SINGULAR (R=0; IRET=1)")
        
        DUA = self._UA0(X, Y, DD, POT1, POT2, POT3, POT4, C0, C1, compute_derivative)
        for I in range(9):
            U[I] = U[I] - DUA[I]
        for I in range(3):
            U[I+9] = U[I+9] + DUA[I+9]



        # IMAGE-SOURCE CONTRIBUTION
        DD = DEPTH - Z
        C1.DCCON1(X, Y, DD, C0)

        DUA = self._UA0(X, Y, DD, POT1, POT2, POT3, POT4, C0, C1, compute_derivative)
        DUB = self._UB0(X, Y, DD, Z, POT1, POT2, POT3, POT4, C0, C1, compute_derivative)
        DUC = self._UC0(X, Y, DD, Z, POT1, POT2, POT3, POT4, C0, C1, compute_derivative)

        for I in range (N_variable):
            DU = DUA[I] + DUB[I] + Z * DUC[I]
            if I >= 10:
                DU = DU + DUC[I-9]
            U[I] = U[I] + DU

        return U
    


    def _UA0(self, X, Y, D, POT1, POT2, POT3, POT4, C0, C1, compute_derivative):
        """
        DISPLACEMENT AND STRAIN AT DEPTH (PART-A)
        DUE TO BURIED POINT SOURCE IN A SEMIINFINITE MEDIUM

        INPUT
        X,Y,D : STATION COORDINATES IN FAULT SYSTEM
        POT1-POT4 : STRIKE-, DIP-, TENSILE- AND INFLATE-POTENCY
        OUTPUT
        U(12) : DISPLACEMENT AND THEIR DERIVATIVES
        """

        # Initialization
        N_variable = 12 if compute_derivative else 3
        U = [torch.zeros_like(X) for _ in range(N_variable)]
        DU = [torch.zeros_like(X) for _ in range(N_variable)]


        ALP1, ALP2, SD, CD = C0.ALP1, C0.ALP2, C0.SD, C0.CD
        P, Q, S, T, XY, X2, R3, QR = C1.P, C1.Q, C1.S, C1.T, C1.XY, C1.X2, C1.R3, C1.QR


        if compute_derivative:
            S2D, C2D = C0.S2D, C0.C2D
            R5, QRX, A3, A5, B3, C3 = C1.R5, C1.QRX, C1.A3, C1.A5, C1.B3, C1.C3 
            UY, VY, WY, UZ, VZ, WZ = C1.UY, C1.VY, C1.WY, C1.UZ, C1.VZ, C1.WZ   


        # STRIKE-SLIP CONTRIBUTION
        if (POT1 != 0.0):
            DU[ 0] =  ALP1 * Q / R3      + ALP2 * X2    * QR
            DU[ 1] =  ALP1 * X / R3 * SD + ALP2 * XY    * QR
            DU[ 2] = -ALP1 * X / R3 * CD + ALP2 * X * D * QR

            if compute_derivative:
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
        if (POT2 != 0.0):
            DU[ 0] =                  ALP2 * X * P * QR
            DU[ 1] =  ALP1 * S / R3 + ALP2 * Y * P * QR
            DU[ 2] = -ALP1 * T / R3 + ALP2 * D * P * QR

            if compute_derivative:
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
        if (POT3 != 0.0):
            DU[ 0] = ALP1 * X / R3 - ALP2 * X * Q * QR
            DU[ 1] = ALP1 * T / R3 - ALP2 * Y * Q * QR
            DU[ 2] = ALP1 * S / R3 - ALP2 * D * Q * QR

            if compute_derivative:
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
        if (POT4 != 0.0):
            DU[ 0] = -ALP1 * X / R3
            DU[ 1] = -ALP1 * Y / R3
            DU[ 2] = -ALP1 * D / R3

            if compute_derivative:
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



    def _UB0(self, X, Y, D, Z, POT1, POT2, POT3, POT4, C0, C1, compute_derivative):
        """
        DISPLACEMENT AND STRAIN AT DEPTH (PART-B)
        DUE TO BURIED POINT SOURCE IN A SEMIINFINITE MEDIUM

        INPUT
        X,Y,D,Z : STATION COORDINATES IN FAULT SYSTEM
        POT1-POT4 : STRIKE-, DIP-, TENSILE- AND INFLATE-POTENCY
        OUTPUT
        U(12) : DISPLACEMENT AND THEIR DERIVATIVES
        """

        # Initialization
        N_variable = 12 if compute_derivative else 3
        U = [torch.zeros_like(X) for _ in range(N_variable)]
        DU = [torch.zeros_like(X) for _ in range(N_variable)]

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


        if compute_derivative:
            D2, R5, QRX, A3, A5, B3, C3 = C1.D2, C1.R5, C1.QRX, C1.A3, C1.A5, C1.B3, C1.C3 
            UY, VY, WY, UZ, VZ, WZ = C1.UY, C1.VY, C1.WY, C1.UZ, C1.VZ, C1.WZ
            D53 = D12 * (8.0 * R2 + 9.0 * R * D + 3.0 * D2) / (R2**2 * RD)
            D54 = D12 * (5.0 * R2 + 4.0 * R * D + D2) / R3 * D12
            FJ1 = -3.0 * XY * (D33 - X2 * D54)
            FJ2 = 1.0 / R3 - 3.0 * D12 + 3.0 * X2 * Y2 * D54
            FK2 = -X * (D32 - Y2 * D53)
            FJ3 = A3 / R3 - FJ2
            FJ4 = -3.0 * XY / R5 - FJ1
            FK1 = -Y * (D32 - X2 * D53)
            FK3 = -3.0 * X * D / R5 - FK2



        # STRIKE-SLIP CONTRIBUTION
        if (POT1 != 0.0) :
            DU[ 0] = -X2 * QR    - ALP3 * FI1 * SD
            DU[ 1] = -XY * QR    - ALP3 * FI2 * SD
            DU[ 2] = -C * X * QR - ALP3 * FI4 * SD

            if compute_derivative:
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
        if (POT2 != 0.0):
            DU[ 0] = -X * P * QR + ALP3 * FI3 * SDCD
            DU[ 1] = -Y * P * QR + ALP3 * FI1 * SDCD
            DU[ 2] = -C * P * QR + ALP3 * FI5 * SDCD

            if compute_derivative:
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
        if (POT3 != 0.0):
            DU[ 0] = X * Q * QR - ALP3 * FI3 * SDSD
            DU[ 1] = Y * Q * QR - ALP3 * FI1 * SDSD
            DU[ 2] = C * Q * QR - ALP3 * FI5 * SDSD

            if compute_derivative:
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
        if (POT4 != 0.0):
            DU[ 0] = ALP3 * X / R3
            DU[ 1] = ALP3 * Y / R3
            DU[ 2] = ALP3 * D / R3

            if compute_derivative:
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



    def _UC0(self, X, Y, D, Z, POT1, POT2, POT3, POT4, C0, C1, compute_derivative):
        """
        DISPLACEMENT AND STRAIN AT DEPTH (PART-C)
        DUE TO BURIED POINT SOURCE IN A SEMIINFINITE MEDIUM

        INPUT
        X,Y,D,Z : STATION COORDINATES IN FAULT SYSTEM
        POT1-POT4 : STRIKE-, DIP-, TENSILE- AND INFLATE-POTENCY
        OUTPUT
        U(12) : DISPLACEMENT AND THEIR DERIVATIVES
        """

        # Initialization
        N_variable = 12 if compute_derivative else 3
        U = [torch.zeros_like(X) for _ in range(N_variable)]
        DU = [torch.zeros_like(X) for _ in range(N_variable)]


        ALP4, ALP5, SD, CD, SDSD, SDCD, S2D, C2D = C0.ALP4, C0.ALP5, C0.SD, C0.CD, C0.SDSD, C0.SDCD, C0.S2D, C0.C2D
        P, Q, S, T = C1.P, C1.Q, C1.S, C1.T
        R2, R3, R5, QR, QRX, A3, A5, C3 = C1.R2, C1.R3, C1.R5, C1.QR, C1.QRX, C1.A3, C1.A5, C1.C3 

        C = D + Z
        QR5 = 5.0 * Q / R2


        if compute_derivative:
            XY, X2, Y2, D2, R = C1.XY, C1.X2, C1.Y2, C1.D2, C1.R
            R7 = R**7
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
        if (POT1 != 0.0):
            DU[ 0] = -ALP4 * A3 / R3 * CD + ALP5 * C * QR * A5
            DU[ 1] = 3.0 * X / R5 * ( ALP4 * Y * CD + ALP5 * C * (SD - Y * QR5))
            DU[ 2] = 3.0 * X / R5 * (-ALP4 * Y * SD + ALP5 * C * (CD + D * QR5))

            if compute_derivative:
                DU[ 3] = ALP4 * 3.0 * X / R5 * (2.0 + A5) * CD - ALP5 * C * QRX * (2.0 + A7)
                DU[ 4] = 3.0 / R5 * ( ALP4 * Y * A5 * CD + ALP5 * C * (A5 * SD - Y * QR5 * A7))
                DU[ 5] = 3.0 / R5 * (-ALP4 * Y * A5 * SD + ALP5 * C * (A5 * CD + D * QR5 * A7))
                DU[ 6] = DU[4]
                DU[ 7] = 3.0 * X / R5 * ( ALP4 * B5 * CD - ALP5 * 5.0 * C / R2 * (2.0 * Y * SD + Q * B7))
                DU[ 8] = 3.0 * X / R5 * (-ALP4 * B5 * SD + ALP5 * 5.0 * C / R2 * (D * B7 * SD - Y * C7 * CD))
                DU[ 9] = 3.0 / R5 * (-ALP4 * D * A5 * CD + ALP5 * C * (A5 * CD + D * QR5 * A7))
                DU[10] = 15.0 * X / R7 * ( ALP4 * Y * D * CD + ALP5 * C * (D * B7 * SD - Y * C7 * CD))
                DU[11] = 15.0 * X / R7 * (-ALP4 * Y * D * SD + ALP5 * C * (2.0* D * CD - Q * C7))

            for I in range(N_variable):
                U[I] = U[I] + POT1 / PI2 * DU[I]


        # DIP-SLIP CONTRIBUTION
        if (POT2 != 0.0):
            DU[ 0] =  ALP4 * 3.0 * X * T / R5              - ALP5 * C * P * QRX
            DU[ 1] = -ALP4 / R3 * (C2D - 3.0 * Y * T / R2) + ALP5 * 3.0 * C / R5 * (S - Y * P * QR5)
            DU[ 2] = -ALP4 * A3 / R3 * SDCD                + ALP5 * 3.0 * C / R5 * (T + D * P * QR5)

            if compute_derivative:
                DU[ 3] = ALP4 * 3.0 * T / R5 * A5                        - ALP5 * 5.0 * C * P * QR / R2 * A7
                DU[ 4] = 3.0 * X / R5 * (ALP4 * (C2D - 5.0 * Y * T / R2) - ALP5 * 5.0 * C / R2 * (S - Y * P * QR7))
                DU[ 5] = 3.0 * X / R5 * (ALP4 * (2.0 + A5) * SDCD        - ALP5 * 5.0 * C / R2 * (T + D * P * QR7))
                DU[ 6] = DU[4]
                DU[ 7] = 3.0 / R5 *     ( ALP4 * (2.0 * Y * C2D + T * B5)      + ALP5 * C * (S2D - 10.0 * Y * S / R2 - P * QR5 * B7))
                DU[ 8] = 3.0 / R5 *     ( ALP4 * Y * A5 * SDCD                 - ALP5 * C * ((3.0 + A5) * C2D + Y * P * DR5 * QR7))
                DU[ 9] = 3.0 * X / R5 * (-ALP4 * (S2D - T * DR5)               - ALP5 * 5.0 * C / R2 * (T + D * P * QR7))
                DU[10] = 3.0 / R5 *     (-ALP4 * (D * B5 * C2D + Y * C5 * S2D) - ALP5 * C * ((3.0 + A5) * C2D + Y * P * DR5 * QR7))
                DU[11] = 3.0 / R5 *     (-ALP4 * D * A5 * SDCD                 - ALP5 * C * (S2D- 10.0 * D * T / R2 + P * QR5 * C7))

            for I in range(N_variable):
                U[I] = U[I] + POT2 / PI2 * DU[I]


        # TENSILE-FAULT CONTRIBUTION
        if (POT3 != 0.0):
            DU[ 0] = 3.0 * X / R5 * (-ALP4 * S + ALP5 * (C * Q * QR5 - Z))
            DU[ 1] =  ALP4 / R3 * (S2D - 3.0 * Y * S / R2) + ALP5 * 3.0 / R5 * (C * (T - Y + Y * Q * QR5) - Y * Z)
            DU[ 2] = -ALP4 / R3 * (1.0 - A3 * SDSD)        - ALP5 * 3.0 / R5 * (C * (S - D + D * Q * QR5) - D * Z)

            if compute_derivative:
                DU[ 3] = -ALP4 * 3.0 * S / R5 * A5 + ALP5 * (C * QR * QR5 * A7 - 3.0 * Z / R5 * A5)
                DU[ 4] = 3.0 * X / R5 * (-ALP4 * (S2D - 5.0 * Y * S / R2)     - ALP5 * 5.0 / R2 * (C * (T - Y+ Y * Q * QR7) - Y * Z))
                DU[ 5] = 3.0 * X / R5 * ( ALP4 * (1.0 - (2.0 + A5) * SDSD)    + ALP5 * 5.0 / R2 * (C * (S - D + D * Q * QR7) - D * Z))
                DU[ 6] = DU[4]
                DU[ 7] = 3.0 / R5 *     (-ALP4 * (2.0 * Y * S2D + S * B5)     - ALP5 * (C * (2.0 * SDSD + 10.0 * Y * (T - Y) / R2 - Q * QR5 * B7) + Z * B5))
                DU[ 8] = 3.0 / R5 *     ( ALP4 * Y * (1.0 - A5 * SDSD)        + ALP5 * (C * (3.0 + A5) * S2D - Y * DR5 * (C * D7 + Z)))
                DU[ 9] = 3.0 * X / R5 * (-ALP4 * (C2D+ S * DR5)               + ALP5 * (5.0 * C / R2 * (S - D + D * Q *QR7) - 1.0 - Z * DR5))
                DU[10] = 3.0 / R5 *     ( ALP4 * (D * B5 * S2D - Y* C5 * C2D) + ALP5 * (C * ((3.0 + A5) * S2D - Y * DR5 * D7) - Y * (1.0 + Z * DR5)))
                DU[11] = 3.0 / R5 *     (-ALP4 * D * (1.0 - A5 * SDSD)        - ALP5 * (C * (C2D+ 10.0 * D * (S - D) / R2 - Q * QR5 * C7) + Z * (1.0 + C5)))

            for I in range(N_variable):
                U[I] = U[I] + POT3 / PI2 * DU[I]


        # INFLATE SOURCE CONTRIBUTION
        if (POT4 != 0.0):
            DU[ 0] = ALP4 * 3.0 * X * D / R5
            DU[ 1] = ALP4 * 3.0 * Y * D / R5
            DU[ 2] = ALP4 * C3 / R3

            if compute_derivative:
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






class DC3D:
    def __init__(self):
        pass
        
        

    def compute(self, ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3, compute_derivative):
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
        
        if (Z > 0.0):
            raise ValueError("POSITIVE Z WAS GIVEN (Z>0; IRET=2)")


        # Initialization
        N_variable = 12 if compute_derivative else 3
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
            ) ):
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
                DUA = self._UA(XI[J], ET[K], Q, DISL1, DISL2, DISL3, C0, C2)

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
            ) ):
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
                DUA = self._UA(XI[J], ET[K], Q, DISL1, DISL2, DISL3, C0, C2)
                DUB = self._UB(XI[J], ET[K], Q, DISL1, DISL2, DISL3, C0, C2)
                DUC = self._UC(XI[J], ET[K] ,Q, Z, DISL1, DISL2, DISL3, C0, C2)

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



    def _UA(self, XI, ET, Q, DISL1, DISL2, DISL3, C0, C2, compute_derivative):
        """
        DISPLACEMENT AND STRAIN AT DEPTH (PART-A)
        DUE TO BURIED FINITE FAULT IN A SEMIINFINITE MEDIUM

        INPUT
            XI,ET,Q : STATION COORDINATES IN FAULT SYSTEM
            DISL1-DISL3 : STRIKE-, DIP-, TENSILE-DISLOCATIONS
        OUTPUT
            U(12) : DISPLACEMENT AND THEIR DERIVATIVES
        """

        # Initialization
        N_variable = 12 if compute_derivative else 3
        U = [torch.zeros_like(XI) for _ in range(N_variable)]
        DU = [torch.zeros_like(XI) for _ in range(N_variable)]

        ALP1, ALP2 = C0.ALP1, C0.ALP2
        R, TT, ALX, ALE, X11, Y11 = C2.R, C2.TT, C2.ALX, C2.ALE, C2.X11, C2.Y11
        QX = Q * X11
        QY = Q * Y11


        if compute_derivative:
            SD, CD = C0.SD, C0.CD
            XI2, Q2, R3, Y, D, Y32 = C2.XI2, C2.Q2, C2.R3, C2.Y, C2.D, C2.Y32
            EY, EZ, FY, FZ, GY, GZ, HY, HZ = C2.EY, C2.EZ, C2.FY, C2.FZ, C2.GY, C2.GZ, C2.HY, C2.HZ
            XY = XI * Y11


        # STRIKE-SLIP CONTRIBUTION
        if (DISL1 != 0.0):
            DU[ 0] = TT / 2.0   + ALP2 * XI * QY
            DU[ 1] =              ALP2 * Q / R
            DU[ 2] = ALP1 * ALE - ALP2 * Q * QY

            if compute_derivative:
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
        if (DISL2 != 0.0):
            DU[ 0] =              ALP2 * Q / R
            DU[ 1] = TT / 2.0   + ALP2 * ET * QX
            DU[ 2] = ALP1 * ALX - ALP2 * Q * QX

            if compute_derivative:
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
        if (DISL3 != 0.0):
            DU[ 0] = -ALP1 * ALE - ALP2 * Q * QY
            DU[ 1] = -ALP1 * ALX - ALP2 * Q * QX
            DU[ 2] =  TT / 2.0   - ALP2 * (ET * QX + XI * QY)

            if compute_derivative:
                DU[ 3] = -ALP1 * XY                  + ALP2 * XI * Q2 * Y32
                DU[ 4] = -ALP1 / R                   + ALP2 * Q2 / R3
                DU[ 5] = -ALP1 * QY                  - ALP2 * Q * Q2 *Y32
                DU[ 6] = -ALP1 * (CD / R + QY * SD)  - ALP2 * Q * FY
                DU[ 7] = -ALP1 * Y * X11             - ALP2 * Q * GY
                DU[ 8] =  ALP1 * (D * X11 + XY * SD) + ALP2 * Q * HY
                DU[ 9] =  ALP1 * (SD / R - QY * CD)  - ALP2 * Q * FZ
                DU[10] =  ALP1 * D * X11             - ALP2 * Q * GZ
                DU[11] =  ALP1 * (Y * X11 + XY * CD) + ALP2 * Q * HZ

            for I in range(N_variable):
                U[I] = U[I] + DISL3 / PI2 * DU[I]


        return U



    def _UB(self, XI, ET, Q, DISL1, DISL2, DISL3, C0, C2, compute_derivative):
        """
        DISPLACEMENT AND STRAIN AT DEPTH (PART-B)
        DUE TO BURIED FINITE FAULT IN A SEMIINFINITE MEDIUM

        INPUT
            XI,ET,Q : STATION COORDINATES IN FAULT SYSTEM
            DISL1-DISL3 : STRIKE-, DIP-, TENSILE-DISLOCATIONS
        OUTPUT
            U(12) : DISPLACEMENT AND THEIR DERIVATIVES
        """

        # Initialization
        N_variable = 12 if compute_derivative else 3
        U = [torch.zeros_like(XI) for _ in range(N_variable)]
        DU = [torch.zeros_like(XI) for _ in range(N_variable)]

        ALP3, SD, CD, SDSD, CDCD, SDCD = C0.ALP3, C0.SD, C0.CD, C0.SDSD, C0.CDCD, C0.SDCD
        XI2, Q2, R, Y, D, TT = C2.XI2, C2.Q2, C2.R, C2.Y, C2.D, C2.TT
        ALE, X11, Y11 = C2.ALE, C2.X11, C2.Y11
            
        RD = R + D

        if (CD != 0.0):
            if (XI == 0.0):
                AI4 = 0.0
            else:
                X = torch.sqrt(XI2 + Q2)
                AI4 = 1.0 / CDCD * (XI / RD * SDCD + 2.0 * torch.atan(
                    (ET * (X + Q * CD) + X * (R + X) * SD) / (XI * (R + X) * CD) 
                ))

            AI3 = (Y * CD / RD - ALE + SD * torch.log(RD)) / CDCD
        else:
            RD2 = RD**2
            AI3 = (ET / RD + Y * Q / RD2 - ALE) / 2.0
            AI4 = XI * Y / RD2 / 2.0

        AI1 = -XI / RD * CD - AI4 * SD 
        AI2 = torch.log(RD) + AI3 * SD 
        QX = Q * X11
        QY = Q * Y11 


        if compute_derivative:
            R3, Y32 = C2.R3, C2.Y32
            EY, EZ, FY, FZ, GY, GZ, HY, HZ = C2.EY, C2.EZ, C2.FY, C2.FZ, C2.GY, C2.GZ, C2.HY, C2.HZ

            D11 = 1.0 / (R * RD)
            AJ2 = XI * Y / RD * D11
            AJ5 = -(D + Y**2 / RD) * D11

            if (CD != 0.0):
                AK1 = XI * (D11 - Y11 * SD) / CD
                AK3 = (Q * Y11 - Y * D11) / CD
                AJ3 = (AK1 - AJ2 * SD) / CD
                AJ6 = (AK3 - AJ5 * SD) / CD
            else:
                AK1 = XI * Q / RD * D11
                AK3 = SD / RD * (XI2 * D11 - 1.0)
                AJ3 = -XI / RD2 * (Q2 * D11 - 0.5)
                AJ6 = - Y / RD2 * (XI2 * D11 - 0.5)

            XY = XI * Y11
            AK2 = 1.0 / R + AK3 * SD
            AK4 = XY * CD - AK1 * SD
            AJ1 = AJ5 * CD - AJ6 * SD
            AJ4 = -XY - AJ2 * CD + AJ3 * SD



        # STRIKE-SLIP CONTRIBUTION
        if (DISL1 != 0.0):
            DU[ 0] = -XI * QY - TT - ALP3 * AI1 * SD
            DU[ 1] = -Q / R        + ALP3 * Y / RD * SD
            DU[ 2] =  Q * QY       - ALP3 * AI2 * SD

            if compute_derivative:
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
        if (DISL2 != 0.0):
            DU[ 0] = -Q / R        + ALP3 * AI3 * SDCD
            DU[ 1] = -ET * QX - TT - ALP3 * XI / RD * SDCD
            DU[ 2] =  Q * QX       + ALP3 * AI4 * SDCD

            if compute_derivative:
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
        if (DISL3 != 0.0):
            DU[ 0] = Q * QY                 - ALP3 * AI3 * SDSD
            DU[ 1] = Q * QX                 + ALP3 * XI / RD * SDSD
            DU[ 2] = ET * QX + XI * QY - TT - ALP3 * AI4 * SDSD

            if compute_derivative:
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



    def _UC(self, XI, ET, Q, Z, DISL1, DISL2, DISL3, C0, C2, compute_derivative):
        """
        DISPLACEMENT AND STRAIN AT DEPTH (PART-C)
        DUE TO BURIED FINITE FAULT IN A SEMIINFINITE MEDIUM

        INPUT
            XI,ET,Q,Z   : STATION COORDINATES IN FAULT SYSTEM
            DISL1-DISL3 : STRIKE-, DIP-, TENSILE-DISLOCATIONS
        OUTPUT
            U(12) : DISPLACEMENT AND THEIR DERIVATIVES
        """

        # Initialization
        N_variable = 12 if compute_derivative else 3
        U = [torch.zeros_like(XI) for _ in range(N_variable)]
        DU = [torch.zeros_like(XI) for _ in range(N_variable)]

        ALP4, ALP5, SD, CD  = C0.ALP4, C0.ALP5, C0.SD, C0.CD, 
        XI2, Q2, R, R3, Y, D = C2.XI2, C2.Q2, C2.R, C2.R3, C2.Y, C2.D
        X11, Y11, X32, Y32 = C2.X11, C2.Y11, C2.X32, C2.Y32

        C = D + Z 
        H = Q * CD - Z
        Z32 = SD / R3 - H * Y32
        XY = XI * Y11
        QY = Q * Y11


        if compute_derivative:
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
        if (DISL1 != 0.0):
            DU[ 0] = ALP4 * XY * CD                  - ALP5 * XI * Q * Z32
            DU[ 1] = ALP4 * (CD / R + 2.0 * QY * SD) - ALP5 * C * Q / R3
            DU[ 2] = ALP4 * QY * CD                  - ALP5 * (C * ET / R3 - Z * Y11 + XI2 * Z32)

            if compute_derivative:
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
        if (DISL2 != 0.0):
            DU[ 0] =  ALP4 * CD / R - QY * SD - ALP5 * C * Q / R3
            DU[ 1] =  ALP4 * Y * X11          - ALP5 * C * ET * Q * X32
            DU[ 2] = -D * X11 - XY * SD       - ALP5 * C * (X11 - Q2 * X32)

            if compute_derivative:
                DU[ 3] = -ALP4 * XI / R3 * CD              + ALP5 * C * XI * QR + XI *Q * Y32 * SD
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
        if (DISL3 != 0.0):
            DU[ 0] = -ALP4 * (SD / R + QY * CD)      - ALP5 * (Z * Y11 - Q2 * Z32)
            DU[ 1] =  ALP4 * 2.0 * XY * SD + D * X11 - ALP5 * C * (X11 - Q2 * X32)
            DU[ 2] =  ALP4 * (Y * X11 + XY * CD)     + ALP5 * Q * (C * ET * X32 + XI * Z32)

            if compute_derivative:
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
