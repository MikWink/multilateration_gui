import numpy as np

# Constants
cl = 3e8  # Speed of light [m/s]
A = 6378137  # Earth's semi-major axis [m] (WGS-84)
B = 0.00669438000426  # First eccentricity squared (WGS-84)


def dms2degrees(dms):
    """Converts degrees, minutes, seconds to decimal degrees."""
    degrees, minutes, seconds = dms
    return degrees + minutes / 60 + seconds / 3600


def w2k(fi, la, h):
    """Converts WGS-84 coordinates (lat, lon, height) to Cartesian (x, y, z)."""
    fi_rad, la_rad = np.radians([fi, la])
    N = A / np.sqrt(1 - B * np.sin(fi_rad) ** 2)
    X = (N + h) * np.cos(fi_rad) * np.cos(la_rad)
    Y = (N + h) * np.cos(fi_rad) * np.sin(la_rad)
    Z = (N * (1 - B) + h) * np.sin(fi_rad)
    return X, Y, Z


def k2w(x, y, z):
    """Converts Cartesian coordinates (x, y, z) to WGS-84 (lat, lon, height)."""
    p = np.sqrt(x ** 2 + y ** 2)
    t = np.arctan2(1.00336408982811 * z, p)
    fi = np.arctan2((z + 42841.31160397718 * np.sin(t) ** 3), (p - 42697.67279723619 * np.cos(t) ** 3))
    la = np.arctan2(y, x)
    h = p / np.cos(fi) - A / np.sqrt(1 - B * np.sin(fi) ** 2)
    return np.degrees(fi), np.degrees(la), h


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

    print(f'A1: {A1}\nA2: {A2}\nA3: {A3}\nA4: {A4}\n')
    print(f'B1: {B1}\nB2: {B2}\nB3: {B3}\nB4: {B4}\nB5: {B5}\nB6: {B6}\nB7: {B7}\n')

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

    print(f'C1: {C1}\nC2: {C2}\nC3: {C3}\nC4: {C4}\nC5: {C5}\nC6: {C6}\nC7: {C7}\nC8: {C8}\nC9: {C9}\nC10: {C10}\n')

    D1 = -A ** 2 - C ** 2
    D2 = -2 * (A * B + C * D)
    D3 = 1 - B ** 2 - D ** 2

    print(f'D1: {D1}\nD2: {D2}\nD3: {D3}\n')

    H = C9 + C5 * A + C6 * C
    I = C5 * B + C6 * D
    E = C10 - C1 * (A ** 2) - C2 * (C ** 2) - C3 * D1 - C4 * A * C - C7 * A - C8 * C
    F = -2 * C1 * A * B - 2 * C2 * C * D - C3 * D2 - C4 * A * D - C4 * C * B - C7 * B - C8 * D
    G = -C1 * B ** 2 - C2 * D ** 2 - C3 * D3 - C4 * B * B

    print(f"H: {H}\nI: {I}\nE: {E}\nF: {F}\nG: {G}\n")

    M1 = G ** 2 - D3 * I ** 2
    M2 = 2 * F * G - D2 * I ** 2 - 2 * D3 * H * I
    M3 = F ** 2 + 2 * E * G - D1 * I ** 2 - 2 * D2 * H * I - D3 * H ** 2
    M4 = 2 * E * F - 2 * D1 * H * I - D2 * H ** 2
    M5 = E ** 2 - D1 * H ** 2

    print(f"M1: {M1}\nM2: {M2}\nM3: {M3}\nM4: {M4}\nM5: {M5}\n")


def pol4(A4, A3, A2, A1, A0):
    """Solves a 4th-degree polynomial equation."""

    # Normalize coefficients
    a4, a3, a2, a1, a0 = A4 / A4, A3 / A4, A2 / A4, A1 / A4, A0 / A4

    # Calculate coefficients of the auxiliary polynomial
    aa2 = a2
    aa1 = a3 * a1 - 4 * a0
    aa0 = 4 * a2 * a0 - a1 ** 2 - a3 ** 2 * a0

    # ... (rest of the polynomial root calculation)


def tdoah():
    """Main TDOA-Hyperbolic algorithm demonstration."""

    # Coordinates of locations (WGS-84)
    locations = {
        "P0": (dms2degrees([49, 39, 20]), dms2degrees([12, 31, 26]), 700),
        "P1": (dms2degrees([49, 48, 43]), dms2degrees([12, 27, 55]), 600),
        "P2": (dms2degrees([49, 32, 28]), dms2degrees([12, 35, 39]), 600),
    }

    # Convert location coordinates to Cartesian
    locations_cartesian = {loc: w2k(*coords) for loc, coords in locations.items()}

    # Target coordinates (WGS-84)
    target = dms2degrees([49, 58, 13]), dms2degrees([13, 32, 59]), 5000
    # Convert target coordinates to Cartesian
    target_cartesian = w2k(target[0], target[1], target[2])

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

    r_0_1 = (r_1 - r_0) / cl
    r_0_2 = (r_2 - r_0) / cl

    print(f'r_0_1: {r_0_1}\nr_0_2: {r_0_2}\n')

    # Translate coordinates
    x1 = locations_cartesian["P1"][0] - locations_cartesian["P0"][0]
    y1 = locations_cartesian["P1"][1] - locations_cartesian["P0"][1]
    z1 = locations_cartesian["P1"][2] - locations_cartesian["P0"][2]
    x2 = locations_cartesian["P2"][0] - locations_cartesian["P0"][0]
    y2 = locations_cartesian["P2"][1] - locations_cartesian["P0"][1]
    z2 = locations_cartesian["P2"][2] - locations_cartesian["P0"][2]

    # Calculate rotaion matrix
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

    a = a11 * x1 + a12 * y1 + a13 * z1
    b = a11 * x2 + a12 * y2 + a13 * z2
    c = a21 * x2 + a22 * y2 + a23 * z2

    print('Rotation Matrix: \n', a11, a12, a13, '\n', a21, a22, a23, '\n', a31, a32, a33, '\n')
    print(f'a: {a}\nb: {b}\nc: {c}\n')

    # Calculate A, B, C, D
    A = (a ** 2 - r_0_1 ** 2) / (2 * a)
    B = -r_0_1 / a
    C = (c ** 2 + b ** 2 - 2 * A * b - r_0_2 ** 2) / (2 * c)
    D = (-B * b - r_0_2) / c

    print(f'A: {A}\nB: {B}\nC: {C}\nD: {D}\n')

    U = 6378137 + target[2]
    W = 6356752.31425 + target[2]

    print(f'U: {U}\nW: {W}\n')

    [K, z] = tdoaell(U, U, W, locations_cartesian["P0"][0], locations_cartesian["P0"][1], locations_cartesian["P0"][2],
                     a11, a12, a13, a21, a22, a23, a31, a32, a33, A, B, C, D)


tdoah()
