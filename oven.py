import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C
from atom import atom
class oven:
    def __init__(self, atom = atom(), capillary_radius = 1, capillary_length = 1, num_of_capillaries = 100):
        """
        __init___ initializes the oven object.
        
        :param temp: operating temperature of the oven.
        :param vap_p_consts: constants for the equation to calculate 
            vapor pressure according to `log_10(P [Pa]) = A + 
        """
        self.atom = atom
        self.c_r = capillary_radius # radius of the capillary
        self.c_l = capillary_length # length of the capillary
        self.c_exit_area = np.pi * np.power(capillary_radius, 2)
        self.c_num = num_of_capillaries
        if np.round(self.c_l / self.c_r) == 1:
            self.clausing = 1 # Clausing factor approximates to one
        else:
            self.clausing = (4 * self.c_r) / (3 * self.c_l) # Clausing factor approx. for thin capillaries
        pass
    def p_vap(self, T):
        return np.power(10, self.atom.a - (self.atom.b / T) - (self.atom.c * np.log10(T)))
    def l_mf(self, T):
        return (C.k * self.atom.T) / (np.sqrt(2) * self.atom.sigma * self.p_vap(T))
    def vav(self, T):
        return np.sqrt(8 * C.k * T / (np.mean([self.atom[key].mass for key in self.atom.isotopes.keys()])))
    def vap_density(self, T):
        return C.N_A * self.p_vap(T) * np.mean([self.atom[key].mass / C.atomic_mass for key in self.atom.isotopes.keys()]) / 1000 / C.k / self.atom.T
    def cap_flux(self, T):
        return self.vap_density(T) * self.vav(T) * self.c_exit_area * self.clausing / 4 # atoms / volume
    # FIGURE THIS OUT
    def cap2_flux(self, T):
        return self.p_vap(T) / np.sqrt(2 * np.pi * np.mean([self.atom[key].mass for key in self.atom.isotopes.keys()]) * C.k * T) * self.clausing
    # FIGURE THIS OUT
    def oven_flux(self, T):
        return self.cap_flux(T) * self.c_num # atoms / volume
    
    def plot_vapor_pressure(self, T_min, T_max, phase_trans = False, res = 1000, endpoint = False, ax = None, figsize = (10, 6), **kwargs):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf()
        temps = np.linspace(T_min, T_max, res, endpoint = endpoint)
        ax.plot(temps, self.p_vap(temps + 273.15) / 133.3)
        if phase_trans:
            ax.axvline(self.atom.melt - 273.15, ls = "--")
        ax.set_title(f"Vapor Pressure vs. Temperature for {self.atom.name}")
        ax.set_xlabel(r"Temperature (${^\circ}$C)")
        ax.set_ylabel("Pressure (Torr)")
        ax.set_yscale("log")
        ax.grid()
        return fig, ax
    def plot_mean_free(self, T_min, T_max, phase_trans = False, res = 1000, endpoint = False, ax = None, figsize = (10, 6)):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf() 
        temps = np.linspace(T_min, T_max, res, endpoint = endpoint)
        ax.plot(temps, self.l_mf(temps))
        if phase_trans:
            ax.axvline(self.atom.melt - 273.15, ls = "--")
        ax.set_title(f"Mean Free Path vs. Temperature for {self.atom.name}")
        ax.set_xlabel(r"Temperature (${^\circ}$C)")
        ax.set_ylabel(r"Mean Free Path, $\lambda$ (m)")
        ax.set_yscale("log")
        ax.grid()
        return fig, ax
    def plot_average_vel(self, T_min, T_max, phase_trans = False, res = 1000, endpoint = False, ax = None, figsize = (10, 6)):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf() 
        temps = np.linspace(T_min, T_max, res, endpoint = endpoint)
        ax.plot(temps, self.vav(temps))
        if phase_trans:
            ax.axvline(self.atom.melt - 273.15, ls = "--")
        ax.set_title(f"Average Velocity vs. Temperature for {self.atom.name}")
        ax.set_xlabel(r"Temperature (${^\circ}$C)")
        ax.set_ylabel(r"Average Velocity, $\langle$ v $\rangle$ (m / s$^2$)")
        ax.set_yscale("log")
        ax.grid()
        return fig, ax
    def plot_cap_flux(self, T_min, T_max, phase_trans = False, res = 1000, endpoint = False, ax = None, figsize = (10, 6)):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf()
        temps = np.linspace(T_min, T_max, res, endpoint = endpoint)
        ax.plot(temps, self.cap_flux(temps))
        if phase_trans:
            ax.axvline(self.atom.melt - 273.15, ls = "--")
        ax.set_title(f"Single Capillary Flux vs. Temperature for {self.atom.name}")
        ax.set_xlabel(r"Temperature (${^\circ}$C)")
        ax.set_ylabel(r"Single Capillary Flux (kg / s)")
        ax.set_yscale("log")
        ax.grid()
        return fig, ax
    def plot_cap2_flux(self, T_min, T_max, phase_trans = False, res = 1000, endpoint = False, ax = None, figsize = (10, 6)):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf()
        temps = np.linspace(T_min, T_max, res, endpoint = endpoint)
        ax.plot(temps, self.cap2_flux(temps))
        if phase_trans:
            ax.axvline(self.atom.melt - 273.15, ls = "--")
        ax.set_title(f"Single Capillary Flux vs. Temperature for {self.atom.name}")
        ax.set_xlabel(r"Temperature (${^\circ}$C)")
        ax.set_ylabel(r"Single Capillary Flux (kg / s)")
        ax.set_yscale("log")
        ax.grid()
        return fig, ax
    def plot_oven_flux(self, T_min, T_max, phase_trans = False, res = 1000, endpoint = False, ax = None, figsize = (10, 6)):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf()
        temps = np.linspace(T_min, T_max, res, endpoint = endpoint)
        ax.plot(temps, self.oven_flux(temps))
        if phase_trans:
            ax.axvline(self.atom.melt - 273.15, ls = "--")
        ax.set_title(f"Oven Flux vs. Temperature for {self.atom.name}")
        ax.set_xlabel(r"Temperature (${^\circ}$C)")
        ax.set_ylabel(r"Oven Flux (kg / s)")
        ax.set_yscale("log")
        ax.grid()
        return fig, ax

def main():
    Ca = atom("Ca.csv")
    o = oven(Ca, 1.14e-3 / 2, 48.3e-3 / 2)
    print(o.clausing)
    fig, ax = o.plot_cap_flux(200, 1200, True)
    fig, ax = o.plot_cap2_flux(200, 1200, True)
    plt.show()
    


if __name__ == "__main__":
    main()