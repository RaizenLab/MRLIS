import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.constants as con
matplotlib.use('TkAgg')


def Saturation(dF, wL):
    sat = (con.pi*con.h*con.c*dF)/(3*wL**3)
    return sat



def main():
    ## Experimental Parameters from Literature
    prepLaser = 460.7333e-9 #Sr I: 5s2 1s0 --> 5s5p 1p1 in [m]
    ionLaser = 405.16e-9 #Sr I: 5s5p 1p1 --> 5p2 1D2 (Autoionizing) in [m]
    einsteinA = 2.01e8 # From NIST, [s^-1]

    ## Experimental Parameters from Calculations
    lineWidth = einsteinA/(2*con.pi)
    Isat = Saturation(lineWidth, prepLaser)


    ## Printouts
    print('Saturation Current of Preparation Laser = ' + str(Isat))

if __name__ == "__main__":
    main()