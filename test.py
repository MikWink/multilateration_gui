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
y =  np.array([-12108.47, -7011.44 , 11167.4])
z = np.array([8294.05, 7053.26, -8326.97])

mat = rotation_matrix_from_vectors(y, np.array([1, 0, 0]))

y_rot = np.dot(mat, y)
z_rot = np.dot(mat, z)

# Define the axis of rotation
axis = np.array([1, 0, 0])  # Rotate about the Y-axis

# Define the angle of rotation in radians
angle = angle_between_vectors([0, 1, 0], z_rot)

# Create a rotation object
rotation = R.from_rotvec(angle * axis)

# Apply the rotation to the vector
x_rot = rotation.apply(x)
y_rot = rotation.apply(y_rot)
z_rot = rotation.apply(z_rot)


print(f"Angle of rotation: {np.degrees(angle)}")

print("Original Vector:", x, y, z)
print("Rotated Vector:", x_rot, y_rot, z_rot)


