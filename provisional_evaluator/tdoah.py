import numpy as np
from evaluation_functions import k2w

class Tdoah:
    def __init__(self, bs, ms):
        self.bs = bs
        self.ms = ms
        self.ms_h = None
        print(f'BS: {self.bs}\nMS: {self.ms}\n')
        self.cl = np.longdouble(3e8)
        self.A = 6378137
        self.B = 0.00669438000426
        self.C = 0.99330561999574
        self.tdoa01 = None
        self.tdoa02 = None

    def solve(self, ms_h, tdoa01, tdoa02):
        self.tdoa01 = tdoa01
        self.tdoa02 = tdoa02
        self.ms_h = ms_h
        P0 = (self.bs[0])
        P1 = (self.bs[1])
        P2 = (self.bs[2])

        locations = [P0, P1, P2]
        locations_cartesian = {}

        for i, loc in enumerate(locations):
            locations_cartesian[f'P{i}'] = loc

        target = self.ms[0], self.ms[1], self.ms[2]

        target_cartesian = target
        #print(f'loc_cart: {locations_cartesian}\ntarget_cart: {target_cartesian}\n')

        # Calculate TDOA values
        r_0 = np.sqrt((target_cartesian[0] - locations_cartesian["P0"][0]) ** 2 + (
                target_cartesian[1] - locations_cartesian["P0"][1]) ** 2 + (
                              target_cartesian[2] - locations_cartesian["P0"][2]) ** 2)
        r_1 = np.sqrt((target_cartesian[0] - locations_cartesian["P1"][0]) ** 2 + (
                target_cartesian[1] - locations_cartesian["P1"][1]) ** 2 + (
                              target_cartesian[2] - locations_cartesian["P1"][2]) ** 2)
        r_2 = np.sqrt((target_cartesian[0] - locations_cartesian["P2"][0]) ** 2 + (
                target_cartesian[1] - locations_cartesian["P2"][1]) ** 2 + (
                              target_cartesian[2] - locations_cartesian["P2"][2]) ** 2)

        #print(f'r_0: {r_0}\nr_1: {r_1}\nr_2: {r_2}\n')

        t_0 = 1/self.cl * r_0
        t_1 = 1/self.cl * r_1
        t_2 = 1/self.cl * r_2

        #print(f't_0: {t_0}\nt_1: {t_1}\nt_2: {t_2}\n')

        if self.tdoa01 == None and self.tdoa02 == None:
            r_0_1 = ((t_1 - t_0) * self.cl)
            r_0_2 = ((t_2 - t_0) * self.cl)
        else:
            r_0_1 = self.tdoa01
            r_0_2 = self.tdoa02

        #print(f'r_0_1: {r_0_1}\nr_0_2: {r_0_2}\n')

        # Translate coordinates
        x1 = locations_cartesian["P1"][0] - locations_cartesian["P0"][0]
        y1 = locations_cartesian["P1"][1] - locations_cartesian["P0"][1]
        z1 = locations_cartesian["P1"][2] - locations_cartesian["P0"][2]
        x2 = locations_cartesian["P2"][0] - locations_cartesian["P0"][0]
        y2 = locations_cartesian["P2"][1] - locations_cartesian["P0"][1]
        z2 = locations_cartesian["P2"][2] - locations_cartesian["P0"][2]

        #print(f'x1: {x1}\nx2: {x2}\ny1: {y1}\ny2: {y2}\nz1: {z1}\nz2: {z2}\n')

        # Calculate the rotation matrix
        a11 = -x1 / (np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2))
        a12 = -y1 / (np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2))
        a13 = -z1 / (np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2))

        temp_a31 = y2 * z1 - y1 * z2
        temp_a32 = z2 * x1 - x2 * z1
        temp_a33 = x2 * y1 - y2 * x1
        a31 = temp_a31 / (np.sqrt(temp_a31 ** 2 + temp_a32 ** 2 + temp_a33 ** 2))
        a32 = temp_a32 / (np.sqrt(temp_a31 ** 2 + temp_a32 ** 2 + temp_a33 ** 2))
        a33 = temp_a33 / (np.sqrt(temp_a31 ** 2 + temp_a32 ** 2 + temp_a33 ** 2))

        temp_a21 = a32 * a13 - a33 * a12
        temp_a22 = a33 * a11 - a31 * a13
        temp_a23 = a31 * a12 - a32 * a11
        a21 = temp_a21 / (np.sqrt(temp_a21 ** 2 + temp_a22 ** 2 + temp_a23 ** 2))
        a22 = temp_a22 / (np.sqrt(temp_a21 ** 2 + temp_a22 ** 2 + temp_a23 ** 2))
        a23 = temp_a23 / (np.sqrt(temp_a21 ** 2 + temp_a22 ** 2 + temp_a23 ** 2))

        # print(f'a11: {a21}\na12: {a22}\na13: {a23}\n')

        a = a11 * x1 + a12 * y1 + a13 * z1
        b = a11 * x2 + a12 * y2 + a13 * z2
        c = a21 * x2 + a22 * y2 + a23 * z2

        #print(f'a: {a}\nb: {b}\nc: {c}\n')
        #print(f'r_0_1: {r_0_1}\nr_0_2: {r_0_2}\n')
        # Calculate A, B, C, D
        A = (a ** 2 - r_0_1 ** 2) / (2 * a)
        B = -r_0_1 / a
        C = (c ** 2 + b ** 2 - 2 * A * b - r_0_2 ** 2) / (2 * c)
        D = (-B * b - r_0_2) / c

        #print(f'A: {A}\nB: {B}\nC: {C}\nD: {D}\n')
        #print(f'ms_h: {self.ms_h}')
        try:
            U = 6378137.0 + self.ms_h
            W = 6356752.31425 + self.ms_h
        except Exception as e:
            print(f'Error: {e}')
        #print(f'U: {U}\nW: {W}\n')
        try:
            KK, zx = self.tdoaell(U, U, W, locations_cartesian["P0"][0], locations_cartesian["P0"][1],
                             locations_cartesian["P0"][2],
                             a11, a12, a13, a21, a22, a23, a31, a32, a33, A, B, C, D)
        except Exception as e:
            print(f'Error: {e}')

        #print(f'KK: {KK}\n')
        #print(f'zx: {zx}\n')

        # Initialize arrays for calculated values with NaNs
        num_roots = len(KK)
        xx = np.full(num_roots, np.nan)
        yy = np.full(num_roots, np.nan)
        zz = np.full(num_roots, np.nan)
        xxx = np.full(num_roots, np.nan)
        yyy = np.full(num_roots, np.nan)
        zzz = np.full(num_roots, np.nan)
        FI = np.full(num_roots, np.nan)  # Latitude in degrees
        LA = np.full(num_roots, np.nan)  # Longitude in degrees
        H = np.full(num_roots, np.nan)  # Height in meters
        solution_wgs = []
        #print("Calculating positions...")
        # Iterate over the roots and calculate positions
        for i in range(num_roots):
            if KK[i] > 0:
                xx[i] = A + B * KK[i]
                yy[i] = C + D * KK[i]
                zz[i] = zx[i]

                # Transform to the original coordinate system
                xxx[i] = a11 * xx[i] + a21 * yy[i] + a31 * zz[i] + locations_cartesian["P0"][0]
                yyy[i] = a12 * xx[i] + a22 * yy[i] + a32 * zz[i] + locations_cartesian["P0"][1]
                zzz[i] = a13 * xx[i] + a23 * yy[i] + a33 * zz[i] + locations_cartesian["P0"][2]

                # Transform to spherical coordinates (WGS-84)
                FI[i], LA[i], H[i] = k2w(xxx[i], yyy[i], zzz[i])
            # else:  # If KK[i] <= 0, values remain NaN (already initialized)
        #print(f'#########################################################\n#######################  RESULTS  #######################\n#########################################################\n')
        #print(f'xxx: {xxx}\nyyy: {yyy}\nzzz: {zzz}\n')
        #print(f'FI: {FI}\nLA: {LA}\nH: {H}\n')
        solution = []
        #print("Checking for real solutions...")
        for i, e in enumerate(FI):
            coords = (FI[i], LA[i], H[i])
            for coord in coords:
                if coord < 0 or np.isnan(coord):
                    #print(f'No real solution for coords: {coords}')
                    break
                if coord == coords[2]:
                    #print(f'Real solution for coords: {coords}')
                    solution = coords
                    solution_wgs = xxx[i], yyy[i], zzz[i]

        #print(f'FI: {self.deg2dms(solution[0])}\nLA: {self.deg2dms(solution[1])}\nH: {solution[2]}\n')
        #solution_conv = solution[0], solution[1], solution[2]
        #print(f'Solution: {solution_wgs[0]}, {solution_wgs[1]}, {solution_wgs[2]}\n')
        return [solution_wgs[0], solution_wgs[1], solution_wgs[2]]

    def tdoaell(self, a, b, c, xc, yc, zc, a11, a21, a31, a12, a22, a32, a13, a23, a33, A, B, C, D):
        """Calculates TDOA for an ellipsoidal Earth model."""

        A1, A2, A3, A4 = (b ** 2) * (c ** 2), a ** 2 * c ** 2, a ** 2 * b ** 2, a ** 2 * b ** 2 * c ** 2
        B1 = A1
        B2 = 2 * A1 * xc
        B3 = A2
        B4 = 2 * A1 * yc
        B5 = A3
        B6 = 2 * A3 * zc
        B7 = A4 - A1 * (xc ** 2) - A2 * (yc ** 2) - A3 * (zc ** 2)

        #print(f'A1: {A1}\nA2: {A2}\nA3: {A3}\nA4: {A4}\n')
        #print(f'B1: {B1}\nB2: {B2}\nB3: {B3}\nB4: {B4}\nB5: {B5}\nB6: {B6}\nB7: {B7}\n')

        C1 = B1 * a11 ** 2 + B3 * a21 ** 2 + B5 * a31 ** 2
        C2 = B1 * a12 ** 2 + B3 * a22 ** 2 + B5 * a32 ** 2
        C3 = B1 * a13 ** 2 + B3 * a23 ** 2 + B5 * a33 ** 2
        C4 = 2 * (B1 * a11 * a12 + B3 * a21 * a22 + B5 * a31 * a32)
        C5 = 2 * (B1 * a11 * a13 + B3 * a21 * a23 + B5 * a31 * a33)
        C6 = 2 * (B1 * a12 * a13 + B3 * a22 * a23 + B5 * a32 * a33)
        C7 = B2 * a11 + B4 * a21 + B6 * a31
        C8 = B2 * a12 + B4 * a22 + B6 * a32
        C9 = B2 * a13 + B4 * a23 + B6 * a33
        C10 = B7

        #print(f'C1: {C1}\nC2: {C2}\nC3: {C3}\nC4: {C4}\nC5: {C5}\nC6: {C6}\nC7: {C7}\nC8: {C8}\nC9: {C9}\nC10: {C10}\n')

        D1 = -A ** 2 - C ** 2
        D2 = -2 * (A * B + C * D)
        D3 = 1 - B ** 2 - D ** 2

        #print(f'D1: {D1}\nD2: {D2}\nD3: {D3}\n')

        H = C9 + C5 * A + C6 * C
        I = C5 * B + C6 * D
        E = C10 - C1 * (A ** 2) - C2 * (C ** 2) - C3 * D1 - C4 * A * C - C7 * A - C8 * C
        F = -2 * C1 * A * B - 2 * C2 * C * D - C3 * D2 - C4 * A * D - C4 * C * B - C7 * B - C8 * D
        G = -C1 * B ** 2 - C2 * D ** 2 - C3 * D3 - C4 * B * B

        #print(f"H: {H}\nI: {I}\nE: {E}\nF: {F}\nG: {G}\n")

        # Version 1
        M1 = 2.7414562480964996e+54
        M2 = 8.686202044656621e+51
        M3 = -4.403132912749302e+68
        M4 = -1.1473908183017031e+65
        M5 = 5.4796754770303925e+78

        #print(f"Type of: {type(M1)}\nM1: {M1}\nM2: {M2}\nM3: {M3}\nM4: {M4}\nM5: {M5}\n")

        # Version 2
        M1 = float(G ** 2 - D3 * I ** 2)  # Force float64 representation
        M2 = float(2 * F * G - D2 * I ** 2 - 2 * D3 * H * I)
        M3 = float(F ** 2 + 2 * E * G - D1 * I ** 2 - 2 * D2 * H * I - D3 * H ** 2)
        M4 = float(2 * E * F - 2 * D1 * H * I - D2 * H ** 2)
        M5 = float(E ** 2 - D1 * H ** 2)

        #print(f"Type of: {type(M1)}\nM1: {M1}\nM2: {M2}\nM3: {M3}\nM4: {M4}\nM5: {M5}\n")

        K = self.pol4(M1, M2, M3, M4, M5)

        #print(f'E: {type(E)}\n F: {type(F)}\nK: {type(K)}\nG: {type(G)}\n')

        zz = np.array([(E + F * k + G * k ** 2) / (H + I * k) for k in K])

        return K, zz

    def pol4(self, A4, A3, A2, A1, A0):

        # Normalize coefficients (divide by leading coefficient A4)
        a4 = A4 / A4
        a3 = A3 / A4
        a2 = A2 / A4
        a1 = A1 / A4
        a0 = A0 / A4

        # Calculate coefficients of the auxiliary cubic polynomial
        aa2 = a2
        aa1 = a3 * a1 - 4 * a0
        aa0 = 4 * a2 * a0 - a1**2 - a3**2 * a0

        #print(f'aa0: {aa0}\naa1: {aa1}\naa2: {aa2}\n')

        # Find a real root of the cubic polynomial
        discriminant = (-aa2**2/9 + aa1/3)**3 + (-aa2**3/27 + (aa1*aa2)/6 + aa0/2)**2

        # Ensure real root calculation for numerical stability
        x03_real = np.real(aa2/3 + ((discriminant**0.5 - aa0/2 - (aa1*aa2)/6 + aa2**3/27)**(1/3) + (-(discriminant**0.5) - aa0/2 - (aa1*aa2)/6 + aa2**3/27)**(1/3)))

        # Calculate intermediate values
        R = np.sqrt(a3**2 / 4 - a2 + x03_real)
        D = np.sqrt((3 * a3**2) / 4 - R**2 - 2 * a2 + (4 * a3 * a2 - 8 * a1 - a3**3) / (4 * R)) if R != 0 else np.sqrt((3 * a3**2) / 4 - 2 * a2 + 2 * np.sqrt(x03_real**2 - 4 * a0))
        E = np.sqrt((3 * a3**2) / 4 - R**2 - 2 * a2 - (4 * a3 * a2 - 8 * a1 - a3**3) / (4 * R)) if R != 0 else np.sqrt((3 * a3**2) / 4 - 2 * a2 - 2 * np.sqrt(x03_real**2 - 4 * a0))

        # Calculate the roots of the fourth-degree polynomial
        x0 = -a3 / 4 + (R + D) / 2
        x1 = -a3 / 4 + (R - D) / 2
        x2 = -a3 / 4 - (R + E) / 2
        x3 = -a3 / 4 - (R - E) / 2

        return np.array([x0, x1, x2, x3])


