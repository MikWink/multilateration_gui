from math import radians, sin, cos, asin, sqrt

# Earth's radius in meters
R = 6371e3

# Convert coordinates to radians
lat1 = radians(50.686887)
lon1 = radians(10.936072)
lat2 = radians(50.686887000525886)
lon2 = radians(10.936071999970057)

# Haversine formula
dlon = lon2 - lon1
dlat = lat2 - lat1
a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
c = 2 * asin(sqrt(a))

# Calculate horizontal distance
horizontal_distance = R * c


# Altitude difference in meters
alt1 = 150
alt2 = 149.99986367858946
dalt = alt2 - alt1

# Euclidean distance for altitude
vertical_distance = abs(dalt)  # Absolute value ensures positive distance


# Combine horizontal and vertical distance using the Pythagorean theorem
total_distance = sqrt(horizontal_distance**2 + vertical_distance**2)

print(f"Total distance in meters: {total_distance:.2f}")
print(f"Total distance in millimeters: {total_distance * 1000:.2f}")
