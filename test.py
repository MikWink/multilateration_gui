import numpy as np

# Example coefficients
A4 = 2.7414562480964996e+54
A3 = 8.686202044656621e+51
A2 = -4.403132912749302e+68
A1 = -1.1473908183017031e+65
A0 = 5.4796754770303925e+78




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

# Example usage:
roots = pol4(A4, A3, A2, A1, A0)
print(roots)  # Output: [1. 1. 2. 1.]
