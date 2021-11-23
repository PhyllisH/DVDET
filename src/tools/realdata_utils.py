import numpy as np
import re
from xml.dom.minidom import parse
from pyquaternion import Quaternion


def euler2quaternion(yaw, pitch, roll):
    cy = np.cos(yaw * 0.5 * np.pi / 180.0)
    sy = np.sin(yaw * 0.5 * np.pi / 180.0)
    cp = np.cos(pitch * 0.5 * np.pi / 180.0)
    sp = np.sin(pitch * 0.5 * np.pi / 180.0)
    cr = np.cos(roll * 0.5 * np.pi / 180.0)
    sr = np.sin(roll * 0.5 * np.pi / 180.0)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return w, x, y, z


def get_origin_enu(xml_file):
    domTree = parse(xml_file)
    rootNode = domTree.documentElement

    SRS = rootNode.getElementsByTagName("SRS")[0].childNodes[0].data
    SRSOrigin = rootNode.getElementsByTagName("SRSOrigin")[0].childNodes[0].data

    data = re.split(r'[:|,]', SRS)
    lon, lat = float(data[2]), float(data[1])
    z = float(SRSOrigin.split(',')[-1])
    return lon, lat, z


def gps_to_decimal(lat_str):
    pattern = r'[°|′|″]'
    arg = re.split(pattern, lat_str)
    return float(arg[0]) + ((float(arg[1]) + (float(arg[2]) / 60)) / 60)


def gps_to_location(origin, long, lat):
    unit_lat = 6371000 * np.pi * 2 / 360
    unit_long = np.cos(np.deg2rad(origin[1])) * 6371000 * np.pi * 2 / 360
    x = unit_long * (long - origin[0])
    y = unit_lat * (lat - origin[1])
    return x, y


if __name__ == '__main__':
    pass