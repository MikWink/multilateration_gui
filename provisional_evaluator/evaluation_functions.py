import numpy as np
import json


def w2k(fi, la, h):
    """Converts WGS-84 coordinates (lat, lon, height) to Cartesian (x, y, z)."""
    K = np.pi / 180
    f = float(fi) * K
    l = float(la) * K
    h = float(h)
    A = 6378137
    B = 0.00669438000426
    C = 0.99330561999574

    a = np.cos(f)
    b = np.sin(f)
    c = np.cos(l)
    d = np.sin(l)

    n = A / np.sqrt(1 - B * b ** 2)

    X = (n + h) * a * c
    Y = (n + h) * a * d;
    Z = (n * C + h) * b;

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
    p = np.sqrt(X ** 2 + Y ** 2)
    fi_rad = np.arctan2(Z, p * (1 - B))  # Initial approximation

    for _ in range(5):  # Iterate for better accuracy
        N = A / np.sqrt(1 - B * np.sin(fi_rad) ** 2)
        h = p / np.cos(fi_rad) - N
        fi_rad = np.arctan2(Z, p * (1 - B * (N / (N + h))))

    fi = np.degrees(fi_rad)

    return fi, la, h


def deg2dms(deg):
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600

    return d, m, s

def read_json(file_path):
    with open(file_path, 'r') as json_file:
        data = extract_data(json.load(json_file))
        return data

def convert_data(bs_data):
  """Converts the given JSON data into the desired list of lists format."""
  result = []
  for key1, value1 in bs_data.items():
    for key2, value2 in value1.items():
      if key2 != '3':  # Assuming '3' is the value you want to exclude
        result.append([float(value1['1']), float(value1['2']), int(value1['3'])])
  return result

def extract_data(json_data):
  """Extracts data from a JSON file with the given structure.

  Args:
    json_data: The JSON data as a dictionary.

  Returns:
    A list of lists, where each inner list contains three values:
    [value1, value2, value3].
  """

  result = []
  for item in json_data.values():
    values = list(item.values())
    result.append(values)
  return result
