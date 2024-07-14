import numpy as np
from pyproj import CRS, Transformer
from decimal import Decimal
from decimal import getcontext
import cmath

# Constants
cl = np.longdouble(3e8)  # Speed of light [m/s]
A = np.longdouble(6378137)  # Earth's semi-major axis [m] (WGS-84)
B = np.longdouble(0.00669438000426)  # First eccentricity squared (WGS-84)

DEBUG = False



def dms2degrees(dms):
    """Converts degrees, minutes, seconds to decimal degrees."""
    degrees, minutes, seconds = dms
    return degrees + minutes / 60 + seconds / 3600


def w2k(fi, la, h):
    """Converts WGS-84 coordinates (lat, lon, height) to Cartesian (x, y, z)."""
    if DEBUG:
        print(f'W2K:::: type of fi: {type(fi)} # {fi:.40f} # , la: {type(la)} # {la:.40f} # , h: {type(h)} # {h:.40f} # ')
    K = np.pi / 180
    #print(f'W2K:::: K: {K}')

    f = fi * K
    l = la * K

    A = 6378137
    B = 0.00669438000426
    C = 0.99330561999574

    #print(f'W2K:::: f: {f}')

    a = np.cos(f)
    b = np.sin(f)
    c = np.cos(l)
    d = np.sin(l)

    n = A / np.sqrt(1 - B * b ** 2)

    X = (n+h)*a*c
    Y = (n+h)*a*d;
    Z = (n*C+h)*b;

    if DEBUG:
        print(f'W2K::::\n{X}\n{Y}\n{Z}\n')
    return X, Y, Z


def k2w(X, Y, Z):
    """
    Converts Cartesian coordinates (X, Y, Z) to WGS-84
    latitude (fi), longitude (la), and height (h) in degrees and meters.
    """

    # WGS-84 ellipsoid parameters
    A = 6378137.0  # Semi-major axis in meters
    B = 0.00669438002290  # Flattening

    # Longitude Calculation
    la_rad = np.arctan2(Y, X)
    la = np.degrees(la_rad)

    # Latitude and Height Calculation (Iterative Method)
    p = np.sqrt(X**2 + Y**2)
    fi_rad = np.arctan2(Z, p * (1 - B))  # Initial approximation

    for _ in range(5):  # Iterate for better accuracy
        N = A / np.sqrt(1 - B * np.sin(fi_rad)**2)
        h = p / np.cos(fi_rad) - N
        fi_rad = np.arctan2(Z, p * (1 - B * (N / (N + h))))

    fi = np.degrees(fi_rad)

    return fi, la, h

def get_geoid_undulation(lat, lon):
    """Fetches geoid undulation (height difference between ellipsoid and mean sea level)
    for a given latitude and longitude using EGM96 model.

    This is a simplified implementation. In real-world scenarios, you'd likely use a library
    or online service to get accurate geoid data.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        Geoid undulation in meters (approximate).
    """
    # EGM96 coefficients for Nuremberg area (very rough approximation)
    # For accurate results, you'd need a full geoid model or a service like:
    # https://geographiclib.sourceforge.io/cgi-bin/GeoidEval
    n_coeff = 360
    c, s = np.mgrid[-180:180, -90:90] / n_coeff
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    geoid = (50 * np.sin(2 * lat_rad) * np.cos(lon_rad) +
             10 * np.cos(4 * lat_rad)) * np.exp(-(lat_rad ** 2 + lon_rad ** 2))
    return geoid



def pol4(A4, A3, A2, A1, A0):
    """
    Calculates the roots of a fourth-degree polynomial.

    Args:
        A4, A3, A2, A1, A0: Coefficients of the polynomial (float or np.array).

    Returns:
        K: A numpy array containing the roots of the polynomial.
    """

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


def tdoaell(a, b, c, xc, yc, zc, a11, a21, a31, a12, a22, a32, a13, a23, a33, A, B, C, D):
    """Calculates TDOA for an ellipsoidal Earth model."""

    A1, A2, A3, A4 = b ** 2 * c ** 2, a ** 2 * c ** 2, a ** 2 * b ** 2, a ** 2 * b ** 2 * c ** 2
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

    K = pol4(M1, M2, M3, M4, M5)

    #print(f'E: {type(E)}\n F: {type(F)}\nK: {type(K)}\nG: {type(G)}\n')

    zz = np.array([(E + F * k + G * k**2) / (H + I * k) for k in K])



    return K, zz



def dms2deg(d, m, s):
    deg = d + m / 60 + s / 3600
    return deg

def deg2dms(deg):
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600
    return d, m, s




def tdoah(bs, ms):
    """Main TDOA-Hyperbolic algorithm demonstration."""
    getcontext().prec = 50

    """P0 = (dms2degrees([50, 41, 39.1]), dms2degrees([10, 54, 60.0]), 700)
    P1 = (dms2degrees([50, 41, 18.7]), dms2degrees([10, 55, 48.4]), 600)
    P2 = (dms2degrees([50, 41, 20.7]), dms2degrees([10, 56, 25.0]), 600)"""

    P0 = (bs[0][0], bs[1][0], bs[2][0])
    P1 = (bs[0][1], bs[1][1], bs[2][1])
    P2 = (bs[0][2], bs[1][2], bs[2][2])

    print(f'P0:\nfic: {P0[0]}\nlac: {P0[1]}\n\nP1:\nfic: {P1[0]}\nlac: {P1[1]}\n\nP2:\nfir: {P2[0]}\nlar: {P2[1]}\n')

    locations = [P0, P1, P2]

    print(k2w(5621990, 636646, 100))

    """# Coordinates of locations (WGS-84)
    locations = {
        "P0": (dms2degrees([49, 39, 20]), dms2degrees([12, 31, 26]), 700),
        "P1": (dms2degrees([49, 48, 43]), dms2degrees([12, 27, 55]), 600),
        "P2": (dms2degrees([49, 32, 28]), dms2degrees([12, 35, 39]), 600),
    }"""
    """test = dms2deg(Decimal("49"), Decimal("39"), Decimal("20"))
    test = Decimal(test)
    print(f'TEST:::: {test:.50f}, type of test: {type(test)}\n')"""

    locations_cartesian = {}

    for i, loc in enumerate(locations):
        #print(f'Converting: {loc}\n')
        loc = w2k(loc[0], loc[1], loc[2])
        locations_cartesian[f'P{i}'] = loc
        #print(f'Converted: {loc}, type of: {type(loc[0])}\n')

    #print(f'locations: {locations}\n')

    """locations_cartesian = {
        loc: w2k(
            np.longdouble(coords[0]),
            np.longdouble(coords[1]),
            np.longdouble(coords[2])
        )
        for loc, coords in locations.items()
    }"""


    # Target coordinates (WGS-84)
    target = ms[0], ms[1], ms[2]
    print(f'target:\nfit: {target[0]}\nlat: {target[1]}\nalt: {target[2]}\n')
    # Convert target coordinates to Cartesian
    target_cartesian = w2k(target[0], target[1], target[2])
    if DEBUG:
        print("test", k2w(target_cartesian[0], target_cartesian[1], target_cartesian[2]))

    print(f'locations_cartesian: {locations_cartesian}\n')
    print(f'target: {target_cartesian}\n')

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

    t_0 = 1/cl * r_0
    t_1 = 1/cl * r_1
    t_2 = 1/cl * r_2
    if DEBUG:
        print(f't_0: {t_0}\nt_1: {t_1}\nt_2: {t_2}\n')

    r_0_1 = (t_1 - t_0) * cl
    r_0_2 = (t_2 - t_0) * cl

    #print(f'r_0_1: {r_0_1}\nr_0_2: {r_0_2}\n')

    #print(f'type of locations_cartesian: {type(locations_cartesian["P1"][1])}\n')

    #print(f'{locations_cartesian["P1"][0]}\n{locations_cartesian["P0"][0]}')
    #print(f'{locations_cartesian["P1"]}\n')

    # Translate coordinates
    x1 = locations_cartesian["P1"][0] - locations_cartesian["P0"][0]
    y1 = locations_cartesian["P1"][1] - locations_cartesian["P0"][1]
    z1 = locations_cartesian["P1"][2] - locations_cartesian["P0"][2]
    x2 = locations_cartesian["P2"][0] - locations_cartesian["P0"][0]
    y2 = locations_cartesian["P2"][1] - locations_cartesian["P0"][1]
    z2 = locations_cartesian["P2"][2] - locations_cartesian["P0"][2]

    #print(f'type of x1: {type(x1)}\n')
    #print(f'x1: {x1}\ny1: {y1}\nz1: {z1}\nx2: {x2}\ny2: {y2}\nz2: {z2}\n')

    # Calculate rotaion matrix
    a11 = -x1 / (np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2))
    a12 = -y1 / (np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2))
    a13 = -z1 / (np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2))

    #print(f'a11: {a11}\na12: {a12}\na13: {a13}\n')

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

    #print(f'a11: {a21}\na12: {a22}\na13: {a23}\n')

    a = a11 * x1 + a12 * y1 + a13 * z1
    b = a11 * x2 + a12 * y2 + a13 * z2
    c = a21 * x2 + a22 * y2 + a23 * z2

    #print('Rotation Matrix: \n', a11, a12, a13, '\n', a21, a22, a23, '\n', a31, a32, a33, '\n')
    #print(f'a: {a}\nb: {b}\nc: {c}\n')

    # Calculate A, B, C, D
    A = (a ** 2 - r_0_1 ** 2) / (2 * a)
    B = -r_0_1 / a
    C = (c ** 2 + b ** 2 - 2 * A * b - r_0_2 ** 2) / (2 * c)
    D = (-B * b - r_0_2) / c
    if DEBUG:
        print(f'r_0_1: {r_0_1}\nr_0_2: {r_0_2}\n')
        print(f'A: {A}\nB: {B}\nC: {C}\nD: {D}\n')

    U = 6378137 + target[2]
    W = 6356752.31425 + target[2]

    #print(f'U: {U}\nW: {W}\n')

    KK, zx = tdoaell(U, U, W, locations_cartesian["P0"][0], locations_cartesian["P0"][1], locations_cartesian["P0"][2],
                     a11, a12, a13, a21, a22, a23, a31, a32, a33, A, B, C, D)

    #print(f'KK: {KK}\nzx: {zx}\n')

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
    print(f'#########################################################\n#######################  RESULTS  #######################\n#########################################################\n')
    print(f'xxx: {xxx}\nyyy: {yyy}\nzzz: {zzz}\n')
    print(f'FI: {FI}\nLA: {LA}\nH: {H}\n')
    solution = []
    for i, e in enumerate(FI):
        coords = (FI[i], LA[i], H[i])
        for coord in coords:
            if coord < 0 or np.isnan(coord):
                if DEBUG:
                    print(f'No real solution for coords: {coords}')
                break
            if coord == coords[2]:
                print(f'Real solution for coords: {coords}')
                solution = coords

    print(f'FI: {deg2dms(solution[0])}\nLA: {deg2dms(solution[1])}\nH: {solution[2]}\n')




"""tdoah()
realTarget = [3999117.68, 963774.50, 4864491.90]
convRealTarget = k2w(realTarget[0], realTarget[1], realTarget[2])
if DEBUG:
    print(f'k2w: {convRealTarget}')
realTargetDMS = [49.97027778, 13.54972222]
convRealTarget = w2k(realTargetDMS[0], realTargetDMS[1], 5000)
if DEBUG:
    print(convRealTarget)"""
