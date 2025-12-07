import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sympy import meijerg, N
import scipy.constants as con
matplotlib.use('TkAgg')

class Isotope:
    def __init__(self, mass, shift, abundance):
        self.mass = mass
        self.shift = shift
        self.abundance = abundance

def averageV(T,m):
    vav = np.sqrt((2*con.k*T)/(m))
    return vav

def Saturation(dF, wL):
    sat = (con.pi*con.h*con.c*dF)/(3*wL**3)
    return sat

def main():
    ## Experimental Parameters from Literature
    prepLaser = 460.7333e-9 #Sr I: 5s2 1s0 --> 5s5p 1p1 [m]
    ionLaser = 405.16e-9 #Sr I: 5s5p 1p1 --> 5p2 1D2 (Autoionizing) [m]
    einsteinA = 2.01e8 # From NIST, [s^-1]
    d = 0.015 # Diameter of laser beams [m]
    crossBarns = 1.2*5450e6 # Ionization cross-section [Barns]
    crossSi = crossBarns*1e-28 # Ionization cross-section [m^2]
    Toven = 630+273.15 # Temperature of the Oven at the exit aperature of capillaries [K]
    sr84 = Isotope(mass=83.913419 * con.atomic_mass, shift=-270.8e6, abundance=0.0056)

    ## Experimental Parameters from Calculations
    lineWidth = einsteinA/(2*con.pi) # Natural Linewidth of selection/preparation laser [Hz]
    Isat = Saturation(lineWidth, prepLaser) # Saturation intensity of selection/preparation laser [W/m^2]
    w2i = (con.c/ionLaser)*2*con.pi # Angular frequency of ionization laser [rad/s]
    gammaParam = crossSi/(2*con.hbar*w2i*d) # Constant containing parameters of ionization [s^2/(kg m)]
    vel0 = averageV(Toven, sr84.mass) # Average velocity of Sr leaving the oven

    ## Generate Data for Efficiency vs Power
    finPow = 100
    Power = np.linspace(1e-6, finPow, num=1000, endpoint=True)
    ionEff = []
    for p in Power:
        frontTerm = ((gammaParam * p) ** 4) / (16 * np.sqrt(con.pi) * vel0 ** 4)
        tmp = ((gammaParam * p) ** 2) / (4 * vel0 ** 2)
        mG = N(meijerg([[], []], [[-2, -1.5, 0], []], tmp))  # Force numerical evaluation
        nu = frontTerm * mG
        ionEff.append(float(1 - nu))  # Convert to float to ensure numerical value

    plt.figure(figsize=(6.5, 4.875))
    plt.plot(Power, ionEff, color="black", linewidth=2.5)
    plt.xlabel("Laser Power (W)")
    plt.ylabel("Ionization Efficiency")
    plt.title("Ionization Efficiency vs. CW Laser Power for Sr-84")
    plt.show()


    ## Printouts
    print(f'Ionization Efficiency at {finPow:.4g} Watts: {100 * ionEff[-1]:.4g}%')
    print(f'Saturation Current of Preparation Laser = {Isat*0.015**2:.4g}W for 1.5cm Beam')

if __name__ == "__main__":
    main()