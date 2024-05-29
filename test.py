import numpy as np

# Example coefficients
A4 = 2.7414562480964996e+54
A3 = 8.686202044656621e+51
A2 = -4.403132912749302e+68
A1 = -1.1473908183017031e+65
A0 = 5.4796754770303925e+78
a4 = A4 / A4
a3 = A3 / A4
a2 = A2 / A4
a1 = A1 / A4
a0 = A0 / A4
aa2=a2
aa1=a3*a1-4*a0
aa0=4*a2*a0-a1**2-a3**2*a0
coeffs = [1, aa2, aa1, aa0]
roots = np.roots(coeffs)
for root in roots:
    if abs(np.imag(root)) < 1e-5:
        print(f'Root: {np.real(root)}')
        root = np.real(root)
print(f"Roots: {roots}")