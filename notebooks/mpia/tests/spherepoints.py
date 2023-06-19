from astropy.coordinates import EarthLocation,SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import AltAz,ICRS

import numpy as np
import matplotlib.pyplot as plt

def random_altaz(n=100,minimum_alt=30):
    costheta = np.random.uniform(np.cos((90-minimum_alt)*np.pi / 180),1,size=n)
    theta = np.arccos(costheta)
    random_az = np.random.rand(n)*2*np.pi
    random_alt = 0.5*np.pi - theta
    
    return random_alt,random_az
    

def fibonacci_sphere(samples=1000):

    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)

def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    elev = np.arctan2(z,np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                           # phi
    return r, elev, az


def uniform_points(n=100,minimum_alt = 30):
    r = 1
    fraction_of_sphere = 2*np.pi * r**2 * (1 - np.cos(0.5*np.pi - minimum_alt/180*np.pi)) / (4*np.pi)
    n2 = n/ fraction_of_sphere
    print(n2)
    my_points = np.array(fibonacci_sphere(int(n2+0.5)))
    r,alt, phi = cart2sph(*my_points.T)
                         
    #alt = np.pi*0.5 - theta
    az = phi                    

    return alt[alt>minimum_alt*np.pi/180],az[alt>minimum_alt*np.pi/180]
    
def get_xyz(r,theta,phi):
    x = r * np.sin( theta) * np.cos( phi )
    y = r * np.sin( theta) * np.sin( phi )
    z = r * np.cos(theta)

    return x,y,z