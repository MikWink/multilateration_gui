import sympy as sp

# Define symbolic variables
C1, C2, C3, C4, C5, C6, C7, C8, C9, C10 = sp.symbols('C1 C2 C3 C4 C5 C6 C7 C8 C9 C10')
A, B, C, D, K, D1, D2, D3 = sp.symbols('A B C D K D1 D2 D3')

# Assign values to constants
C1 = 1.65332569366 * 10**27
C2 = 1.64905820483 * 10**27
C3 = 1.65573435311 * 10**27
C4 = -9.63348722199 * 10**23
C5 = -1.07853326585 * 10**25
C6 = 1.20222206555 * 10**24
C7 = 1.47386540473 * 10**32
C8 = -2.25909633913 * 10**33
C9 = 2.09564982948 * 10**34
C10 = 9.06676987692 * 10**37

A = -8859.230
B = -0.101289
C = 108408.16
D = -0.358413
D1 = -1.18308160592 * 10**10
D2 = 7.59152816973 * 10**4
D3 = 0.8612800051

# Define the equation
equation = C1 * (A + B*K)**2 + C2 * (C + D*K)**2 + C3 * (D1 + D2*K + D3*K**2)**2 + \
          C4 * (A + B*K) * (C + D*K) + C5 * (A + B*K) * (D1 + D2*K + D3*K**2) + \
          C6 * (C + D*K) * (D1 + D2*K + D3*K**2) + C7 * (A + B*K) + \
          C8 * (C + D*K) + C9 * (D1 + D2*K + D3*K**2) - C10

# Solve for K
K = sp.solve(equation, K)

# Print the solution(s)
if len(K) == 1:
  print("K =", K[0])
else:
  print("There are multiple possible values for K.")
  for k in K:
    print("K =", k)