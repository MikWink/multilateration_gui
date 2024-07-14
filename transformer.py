import numpy as np
from pyproj import Proj


class CoordinateTransformer:

    def __init__(self):
        pass

    @staticmethod
    def utm_to_long(loc, zone_no="32"):
        myProj = Proj("+proj=utm +zone=" + zone_no + " +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        long, lat = myProj(loc[0], loc[1], inverse=True)
        loc = np.array([long, lat])
        return loc

    @staticmethod
    def long_to_utm(loc, zone_no="32"):
        myProj = Proj("+proj=utm +zone=" + zone_no + " +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        utm_x, utm_y = myProj(loc[1], loc[0])
        loc = np.array([utm_x, utm_y])
        return loc
