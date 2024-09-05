import numpy as np

def filter_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

def filter_list(coordinates):
    x_values = np.array(coordinates[0])
    y_values = np.array(coordinates[1])
    z_values = np.array(coordinates[2])

    # Filter each axis
    filtered_x = filter_outliers(x_values)
    filtered_y = filter_outliers(y_values)
    filtered_z = filter_outliers(z_values)

    # Combine filtered coordinates back into a list
    filtered_coordinates = [filtered_x.tolist(), filtered_y.tolist(), filtered_z.tolist()]

    return filtered_coordinates






