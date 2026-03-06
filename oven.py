import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C
from atom import atom
class oven:
    def __init__(self, atom = atom()):
        """
        __init___ initializes the oven object.
        
        :param temp: operating temperature of the oven.
        :param vap_p_consts: constants for the equation to calculate 
            vapor pressure according to `log_10(P [Pa]) = A + 
        """
        self.atom = atom.name
        self.melt = atom.melt
        self.atom_sigma = np.pi * np.power(atom, 2)
        self.a, self.b, self.c = atom.a, atom.b, atom.c
        self.T = atom.T + 273.15
        pass
    def p_vap(self, T):
        return np.power(10, self.a - (self.b / T) - (self.c * np.log10(T))) / 133.3
    def l_mf(self):
        return (C.k * self.T) / (np.sqrt(2) * self.atom_sigma * self.p_vap(self.T))
    
    def plot_mean_free(self, T_min, T_max, phase_trans = False, res = 1000, endpoint = False, ax = None, figsize = (10, 6)):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf() 
        temps = np.linspace(T_min, T_max, res, endpoint = endpoint)
        ax.plot(temps, self.l_mf())
        
    def plot_vapor_pressure(self, T_min, T_max, phase_trans = False, res = 1000, endpoint = False, ax = None, figsize = (10, 6), **kwargs):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf()
        temps = np.linspace(T_min, T_max, res, endpoint = endpoint)
        ax.plot(temps, self.p_vap(temps + 273.15))
        if phase_trans:
            ax.axvline(self.melt - 273.15, ls = "--")
        ax.set_title(f"Temperature vs. Vapor Pressure for {self.atom}")
        ax.set_xlabel(r"Temperature (${^\circ}$C)")
        ax.set_ylabel("Pressure (Torr)")
        ax.set_yscale("log")
        ax.grid()
        return fig, ax

def main():
    Ca = atom("Ca.csv")
    o = oven(Ca)
    fig, ax = o.plot_vapor_pressure(200, 1200, True)
    plt.show()


if __name__ == "__main__":
    main()