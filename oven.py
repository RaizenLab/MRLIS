import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C
from atom import atom
class oven:
    def __init__(self, atom = atom(), capillary_radius = 1.143e-3 / 2, capillary_length = 48.26e-3, num_of_capillaries = 35, grams = 3):
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
        self.beta = 2 * capillary_radius / capillary_length # aspect ratio of capillaries
        self.alpha = 0.5 - (1 / 3 / np.power(self.beta, 2)) * (1 - 2 * np.power(self.beta, 3) + (2 * np.power(self.beta, 2) - 1) * np.sqrt(1 + np.power(self.beta, 2))) / (np.sqrt(1 + np.power(self.beta, 2)) - np.power(self.beta, 2) * np.arcsinh(1 / self.beta)) # rate of wall collisions at the exit plane of the channel
        self.tot_grams = grams
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
        return np.sqrt(8 * C.k * T / (np.sum([self.atom[key].mass * self.atom[key].abund for key in self.atom.isotopes.keys()])) / np.pi)
    def vap_density(self, T):
        return self.p_vap(T) / C.k / self.atom.T
    def cap_flux(self, T):
        return self.vap_density(T) * self.vav(T) * self.c_exit_area * self.clausing / 4 # atoms / volume
    def oven_flux(self, T):
        return self.cap_flux(T) * self.c_num # atoms / volume
    def tot_work_time(self, T):
        return self.tot_grams * C.N_A / (np.sum([self.atom[key].mass * self.atom[key].abund for key in self.atom.isotopes.keys()]) / C.atomic_mass) / self.oven_flux(T) / (60 * 60 * 24 * 30)
    def R(self, theta): # distance from the capillary
        return np.arccos(np.tan(theta) / self.beta) - (np.tan(theta) / self.beta) * np.sqrt(1 - np.power(np.tan(theta) / self.beta, 2))
    def J(self, theta): # angular distribution function
        if np.tan(theta) < self.beta:
            return self.alpha * np.cos(theta) + (2 / np.pi) * np.cos(theta) * ((1 - self.alpha) * self.R(theta) + (2 / 3 / np.tan(theta) * self.beta) * (1 - 2 * self.alpha) * (1 - np.power(1 - np.power(np.tan(theta) / self.beta, 2), (3 / 2))))
        else:
            return self.alpha * np.cos(theta) + (4 / (3 * np.pi * np.tan(theta) / self.beta)) * (1 - 2 * self.alpha) * np.cos(theta)
    
    def plot_vapor_pressure(self, T_min, T_max, phase_trans = False, res = 1000, endpoint = False, ax = None, figsize = (10, 6), save = False, **kwargs):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf()
        temps = np.linspace(T_min, T_max, res, endpoint = endpoint)
        ax.plot(temps, self.p_vap(temps + 273.15) / 133.3)
        if phase_trans:
            ax.axvline(self.atom.melt - 273.15, ls = "--")
        ax.set_title(f"Vapor Pressure vs. Temperature for {self.atom.name}")
        title = f"Vapor Pressure vs. Temperature for {self.atom.name}"
        ax.set_xlabel(r"Temperature (${^\circ}$C)")
        ax.set_ylabel("Pressure (Torr)")
        ax.set_yscale("log")
        ax.grid()
        if save:
            fig.savefig(title + ".png")
        return fig, ax
    def plot_mean_free(self, T_min, T_max, phase_trans = False, res = 1000, endpoint = False, ax = None, figsize = (10, 6), save = False):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf() 
        temps = np.linspace(T_min, T_max, res, endpoint = endpoint)
        ax.plot(temps, self.l_mf(temps + 273.15))
        if phase_trans:
            ax.axvline(self.atom.melt - 273.15, ls = "--")
        ax.set_title(f"Mean Free Path vs. Temperature for {self.atom.name}")
        title = f"Mean Free Path vs. Temperature for {self.atom.name}"
        ax.set_xlabel(r"Temperature (${^\circ}$C)")
        ax.set_ylabel(r"Mean Free Path, $\lambda$ (m)")
        ax.set_yscale("log")
        ax.grid()
        if save:
            fig.savefig(title + ".png")
        return fig, ax
    def plot_average_vel(self, T_min, T_max, phase_trans = False, res = 1000, endpoint = False, ax = None, figsize = (10, 6), save = False):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf() 
        temps = np.linspace(T_min, T_max, res, endpoint = endpoint)
        ax.plot(temps, self.vav(temps + 273.15))
        if phase_trans:
            ax.axvline(self.atom.melt - 273.15, ls = "--")
        ax.set_title(f"Average Velocity vs. Temperature for {self.atom.name}")
        title = f"Average Velocity vs. Temperature for {self.atom.name}"
        ax.set_xlabel(r"Temperature (${^\circ}$C)")
        ax.set_ylabel(r"Average Velocity, $\langle$ v $\rangle$ (m / s)")
        # ax.set_yscale("log")
        ax.grid()
        if save:
            fig.savefig(title + ".png")
        return fig, ax
    def plot_cap_flux(self, T_min, T_max, phase_trans = False, res = 1000, endpoint = False, ax = None, figsize = (10, 6), save = False):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf()
        temps = np.linspace(T_min, T_max, res, endpoint = endpoint)
        ax.plot(temps, self.cap_flux(temps + 273.15))
        if phase_trans:
            ax.axvline(self.atom.melt - 273.15, ls = "--")
        ax.set_title(f"Single Capillary Flux vs. Temperature for {self.atom.name}")
        title = f"Single Capillary Flux vs. Temperature for {self.atom.name}"
        ax.set_xlabel(r"Temperature (${^\circ}$C)")
        ax.set_ylabel(r"Single Capillary Flux (kg / s)")
        ax.set_yscale("log")
        ax.grid()
        if save:
            fig.savefig(title + ".png")
        return fig, ax
    def plot_oven_flux(self, T_min, T_max, phase_trans = False, res = 1000, endpoint = False, ax = None, figsize = (10, 6), save = False):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf()
        temps = np.linspace(T_min, T_max, res, endpoint = endpoint)
        ax.plot(temps, self.oven_flux(temps + 273.15))
        if phase_trans:
            ax.axvline(self.atom.melt - 273.15, ls = "--")
        ax.set_title(f"Oven Flux vs. Temperature for {self.atom.name}")
        title = f"Oven Flux vs. Temperature for {self.atom.name}"
        ax.set_xlabel(r"Temperature (${^\circ}$C)")
        ax.set_ylabel(r"Oven Flux (atoms / s)")
        ax.set_yscale("log")
        ax.grid()
        if save:
            fig.savefig(title + ".png")
        return fig, ax
    def plot_tot_work_time(self, T_min, T_max, phase_trans = False, res = 1000, endpoint = False, ax = None, figsize = (10, 6), save = False):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf()
        temps = np.linspace(T_min, T_max, res, endpoint = endpoint)
        ax.plot(temps, self.tot_work_time(temps + 273.15))
        if phase_trans:
            ax.axvline(self.atom.melt - 273.15, ls = "--")
        ax.set_title(f"Total Oven Working Time vs. Temperature for {self.atom.name}")
        title = f"Total Oven Working Time vs. Temperature for {self.atom.name}"
        ax.set_xlabel(r"Temperature (${^\circ}$C)")
        ax.set_ylabel(r"Oven Working Time (months)")
        ax.set_yscale("log")
        ax.grid()
        if save:
            fig.savefig(title + ".png")
        return fig, ax
    def plot_ang_dist_sing_channel(self, theta_min, theta_max, radians = True, res = 1000, endpoint = False, ax = None, figsize = (10, 6), save = False):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf()
        thetas = np.linspace(theta_min, theta_max, res, endpoint = endpoint)
        thetas = thetas[thetas != 0]
        if radians:
            ax.plot(thetas, [self.J(t) for t in thetas])
        else:
            ax.plot(thetas, [self.J(t / (180 * np.pi)) for t in thetas])
        title = f"Angular Distribution for Single Capillary vs. Half Angle for {self.atom.name}"
        ax.set_title(title)
        if radians:
            ax.set_xlabel(r"Half Angle, $\theta$ (radians)")
        else:
            ax.set_xlabel(r"Half Angle, $\theta$ ($^\circ$)")
        ax.set_ylabel(r"Angular Distribution")
        ax.grid()
        if save:
            fig.savefig(title + ".png")
        return fig, ax

def main():
    Ca = atom("Ca.csv")
    o = oven(Ca)
    o.plot_ang_dist_sing_channel(-0.1, 0.1)
    # fig, ax = o.plot_vapor_pressure(300, 800, True)
    # fig, ax = o.plot_mean_free(300, 800, True)
    # fig, ax = o.plot_average_vel(300, 800, True)
    # fig, ax = o.plot_cap_flux(300, 800, True)
    # fig, ax = o.plot_oven_flux(300, 800, True)
    # fig, ax = o.plot_tot_work_time(300, 800, True)
    plt.show()

if __name__ == "__main__":
    main()