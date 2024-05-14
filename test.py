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
axis = np.array([1, 0, 0])  # Rotate about the Y-axis

# Define the angle of rotation in radians
u = np.subtract([y_rot[0], z_rot[1], z_rot[2]], [y_rot[0], 0, 0])
v = np.subtract([y_rot[0], z_rot[1], 0], [y_rot[0], 0, 0])
print(f"Vectors for angle calculation:\n{u}\n{v}\n")
angle = angle_between_vectors(u, v)
print(f"Angle of rotation:\n{np.degrees(angle)}\n")

# Create a rotation object
rotation = R.from_rotvec(angle * axis)

# Project z_rot onto the yz-plane
z_proj = np.array([0, z_rot[1], z_rot[2]])
# Determine the angle to rotate such that z-component of z_rot becomes zero
angle_z = np.arctan2(z_proj[2], z_proj[1])
# Create the rotation object
rotation_z = R.from_rotvec(angle_z * axis)

# Construct the rotation matrix for the second rotation
rotation_z_mat = np.array([
    [1, 0, 0],
    [0, np.cos(angle_z), -np.sin(angle_z)],
    [0, np.sin(angle_z), np.cos(angle_z)]
])

# Since we need to rotate in the opposite direction to align with the y-axis
rotation_z_mat = rotation_z_mat.T

# Apply the second rotation
z_rot_final = np.dot(rotation_z_mat, z_rot)

# Combine both rotations
final_rotation_matrix = np.dot(rotation_z_mat, mat)

x_rot = x
print(
    f"Second rotation:\n{[round(x_rot[0]), round(x_rot[1]), round(x_rot[2])]}\n{[round(y_rot[0]), round(y_rot[1]), round(y_rot[2])]}\n{[round(z_rot_final[0]), round(z_rot_final[1]), round(z_rot_final[2])]}\n")

print(f"Final Rotation Matrix:\n{final_rotation_matrix}\n")
