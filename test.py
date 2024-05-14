from scipy.spatial.transform import Rotation as R
import numpy as np


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def angle_between_vectors(u, v):
    # Normalize the vectors
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)

    # Calculate the dot product
    dot_product = np.dot(u, v)

    # Calculate the cosine of the angle
    cos_theta = dot_product / (u_norm * v_norm)

    # Convert the cosine to an angle in radians
    theta = np.arccos(cos_theta)

    return theta


# Define the vector to be rotated
x = np.array([0, 0, 0])
y = np.array([-12108.47, -7011.44, 11167.4])
z = np.array([8294.05, 7053.26, -8326.97])

print(f"Original Vector:\n{x}\n{y}\n{z}\n")

mat = rotation_matrix_from_vectors(y, np.array([1, 0, 0]))

print(f"First rotation matrix:\n{mat}\n")

y_rot = np.dot(mat, y)
z_rot = np.dot(mat, z)

print(f"First rotation:\n{x}\n{y_rot}\n{z_rot}\n")

# Define the axis of rotation
axis = np.array([1, 0, 0])

# Define the angle of rotation in radians
u = np.subtract([y_rot[0], z_rot[1], z_rot[2]], [y_rot[0], 0, 0])
v = np.subtract([y_rot[0], z_rot[1], 0], [y_rot[0], 0, 0])
print(f"Vectors for angle calculation:\n{u}\n{v}\n")
angle = angle_between_vectors(u, v)
print(f"Angle of rotation:\n{np.degrees(angle)}")

# Create a rotation object
rotation = R.from_rotvec(angle * axis)

# Project z_rot onto the yz-plane
z_proj = np.array([0, z_rot[1], z_rot[2]])
# Determine the angle to rotate such that z-component of z_rot becomes zero
angle_z = np.arctan2(z_proj[2], z_proj[1])
print(f"Angle of rotation:\n{np.degrees(angle_z)}\n")
# Create the rotation object
rotation_z = R.from_rotvec(angle_z * axis)

# Construct the rotation matrix for the second rotation
rotation_z_mat = np.array([
    [-1, 0, 0],
    [0, np.cos(angle_z), -np.sin(angle_z)],
    [0, np.sin(angle_z), np.cos(angle_z)]
])

# Since we need to rotate in the opposite direction to align with the y-axis
rotation_z_mat = rotation_z_mat.T

print(f"Second Rotation mat:\n{rotation_z_mat}\n")

# Apply the second rotation
z_rot_final = np.dot(rotation_z_mat, z_rot)
y_rot = np.dot(rotation_z_mat, y_rot)

# Combine both rotations
final_rotation_matrix = np.dot(rotation_z_mat, mat.T)



x_rot = x
print(
    f"Second rotation:\n{[round(x_rot[0]), round(x_rot[1]), round(x_rot[2])]}\n{[round(y_rot[0]), round(y_rot[1]), round(y_rot[2])]}\n{[round(z_rot_final[0]), round(z_rot_final[1]), round(z_rot_final[2])]}\n")

print(f"Final Rotation Matrix:\n{final_rotation_matrix.T}\n")

a = y_rot[0]
b = z_rot_final[0]
c = z_rot_final[1]



L = (-6.044 * 10 ** -6) * 300000000
R = (6.916 * 10 ** -6) * 300000000

A = (a**2 - L**2)/(2*a)
B = -L/a
C = (c**2 + b**2 - 2*A*b - R**2)/(2*c)
D = (-B*b - R)/c

print(f"L: = {L}\nR: = {R}\nA: = {A}\nB: = {B}\nC: = {C}\nD: = {D}\n")

h = 5000
semi_minor = 6356752.31425
semi_major = 6378137

U = semi_major+h
V = semi_major+h
W = semi_minor+h

print(f"U: = {U}\nV: = {V}\nW: = {W}\n")

A1 = V**2 * U**2
A2 = U**2 * W**2
A3 = U**2 * V**2
A4 = U**2 * V**2 * W**2

print(f"A1: = {A1}\nA2: = {A2}\nA3: = {A3}\nA4: = {A4}\n")

P0 = [4039139.89, 897222.76, 4838608.59]

B1 = A1
B2 = -2*A1*P0[0]
B3 = A2
B4 = -2*A1*P0[1]
B5 = A3
B6 = -2*A3*P0[2]
B7 = A4 - A1*P0[0]**2 - A2*P0[1]**2 - A3*P0[2]**2

print(f"B1: = {B1}\nB2: = {B2}\nB3: = {B3}\nB4: = {B4}\nB5: = {B5}\nB6: = {B6}\nB7: = {B7}\n")

C1 = B1 * final_rotation_matrix.T[0][0]**2 + B3 * final_rotation_matrix.T[1][0]**2 + B5 * final_rotation_matrix.T[2][0]**2
C2 = B1 * final_rotation_matrix.T[0][1]**2 + B3 * final_rotation_matrix.T[1][1]**2 + B5 * final_rotation_matrix.T[2][1]**2
C3 = B1 * final_rotation_matrix.T[0][2]**2 + B3 * final_rotation_matrix.T[1][2]**2 + B5 * final_rotation_matrix.T[2][2]**2

C4 = 2*(B1*final_rotation_matrix.T[0][0]*final_rotation_matrix.T[0][1] + B3*final_rotation_matrix.T[1][0]*final_rotation_matrix.T[1][1] + B5*final_rotation_matrix.T[2][0]*final_rotation_matrix.T[2][1])
C5 = 2*(B1*final_rotation_matrix.T[0][0]*final_rotation_matrix.T[0][2] + B3*final_rotation_matrix.T[1][0]*final_rotation_matrix.T[1][2] + B5*final_rotation_matrix.T[2][0]*final_rotation_matrix.T[2][2])
C6 = 2*(B1*final_rotation_matrix.T[0][1]*final_rotation_matrix.T[0][2] + B3*final_rotation_matrix.T[1][1]*final_rotation_matrix.T[1][2] + B5*final_rotation_matrix.T[2][1]*final_rotation_matrix.T[2][2])

C7 = B2*final_rotation_matrix.T[0][0] + B4*final_rotation_matrix.T[1][0] + B6*final_rotation_matrix.T[2][0]
C8 = B2*final_rotation_matrix.T[0][1] + B4*final_rotation_matrix.T[1][1] + B6*final_rotation_matrix.T[2][1]
C9 = B2*final_rotation_matrix.T[0][2] + B4*final_rotation_matrix.T[1][2] + B6*final_rotation_matrix.T[2][2]

D1 = -A**2 - C**2
D2 = -2*(A*B + C*D)
D3 = 1-B**2 - D**2

print(f"C1: = {C1}\nC2: = {C2}\nC3: = {C3}\nC4: = {C4}\nC5: = {C5}\nC6: = {C6}\nC7: = {C7}\nC8: = {C8}\nC9: = {C9}\n")
print(f"D1: = {D1}\nD2: = {D2}\nD3: = {D3}\n")

H = C9 + C5*A + C6*C
I = C5*B + C6*D
E = 0
F = -2*C1*A*B - 2*C2*C*D - C3*D2 - C4*A*D - C4*C*B - C7*B - C8*D
G = -C1*B**2 - C2*D**2 - C3*D3 - C4*B*B

print(f"H: = {H}\nI: = {I}\nE: = {E}\nF: = {F}\nG: = {G}\n")

M1 = G**2 - D3*I**2
M2 = 2*F*G - D2*I**2 - 2*D3*H*I
M3 = F**2 + 2*E*G - D1*I**2 - 2*D2*H*I - D3*H**2
M4 = 2*E*F - 2* D1*H*I - D2*H**2
M5 = E**2 - D1*H**2

print(f"M1: = {M1}\nM2: = {M2}\nM3: = {M3}\nM4: = {M4}\nM5: = {M5}\n")