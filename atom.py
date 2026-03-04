import csv
import isotope as iso
import numpy as np
import scipy.constants as C
import matplotlib.pyplot as plt
class atom:
    def __init__(self, f, transition_wl, A, oven_T = 650 + 273.15, theta = 0.02609):
        self.name = f[:-4]
        self.wl = transition_wl
        self.freq = C.c / self.wl
        self.A = A
        self.nat_lw = A / (2 * np.pi)
        self.T = oven_T
        self.theta = theta
        self.isotopes = dict()
        self.headers = []
        with open(f, newline = "") as f:
            r = csv.reader(f, delimiter = ",")
            for row in r:
                if row[0] == "Isotope #":
                    self.headers = row
                    continue
                self.isotopes[row[0]] = iso.isotope(r"$^{" + row[0] + r"}$" + self.name, *list(map(float, row[1:])), C.c / self.wl)
            f.close()
        pass
    def __str__(self):
        return "".join([str(x) for x in list(self.isotopes.values())])
    def __getitem__(self, key):
        if type(key) == str:
            try:
                return self.isotopes[key]
            except KeyError:
                print(f"No data for $^{key}${self.name}")
        if type(key) == int:
            try:
                return self.isotopes[list(self.isotopes.keys())[key]]
            except KeyError:
                print(f"Index {key} is not in {self.name}")
    def __len__(self):
        return len(self.isotopes.keys())
    def plot_spectra(self, f_min, f_max, res = 1000, show_transitions = False, total_spectra = False, indiv_spectra = True, ax = None, figsize = (10, 6)):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf()
        freqs = np.linspace(f_min, f_max, res)
        if total_spectra:
            t_spec = np.zeros_like(freqs)
        for key in self.isotopes.keys():
            if indiv_spectra:
                ax.plot(freqs / 1e9, self[key].spectra(freqs, self[key].freq, self.nat_lw, self[key].doppler(self[key].freq, self[key].average_vel(self.T), self.theta)), label = self[key].name)
            if show_transitions:
                ax.axvline(self[key].freq / 1e9, ls = "--")
                ax.text(self[key].freq / 1e9, 1.01 * np.max(self[key].spectra(freqs, self[key].freq, self.nat_lw, self[key].doppler(self[key].freq, self[key].average_vel(self.T), self.theta))), self[key].name, bbox = dict(facecolor='white', alpha=1, edgecolor='gray', boxstyle='square, pad=0.1'))
            if total_spectra:
                t_spec += self[key].spectra(freqs, self[key].freq, self.nat_lw, self[key].doppler(self[key].freq, self[key].average_vel(self.T), self.theta))
        if total_spectra:
            ax.plot(freqs / 1e9, t_spec, label = "Total Spectra")
        ax.set_title(f"Isotope Spectra for {self.name} Isotopes, Relative to {self[int(np.where(np.array([self[i].shift for i in range(len(self))]) == 0)[0][0])].name}")
        return fig, ax
    def plot_absorp_signal(self, f_min, f_max, res = 1000, show_transitions = False, total_absorp = False, indiv_sig = True, ax = None, figsize = (10, 6)):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf()
        freqs = np.linspace(f_min, f_max, res)
        if total_absorp:
            t_abs = np.zeros_like(freqs)
        for key in self.isotopes.keys():
            if indiv_sig:
                ax.plot(freqs / 1e9, self[key].absorp_sig(freqs, self[key].freq, self.nat_lw, self[key].doppler(self[key].freq, self[key].average_vel(self.T), self.theta)), label = self[key].name)
            if show_transitions:
                ax.axvline(self[key].freq / 1e9, ls = "--")
            if total_absorp:
                t_abs += self[key].absorp_sig(freqs, self[key].freq, self.nat_lw, self[key].doppler(self[key].freq, self[key].average_vel(self.T), self.theta))
        if total_absorp:
            ax.plot(freqs / 1e9, t_abs, label = "Total Spectra")
        ax.set_title(f"Absorption Spectra for {self.name} Isotopes, Relative to {self[int(np.where(np.array([self[i].shift for i in range(len(self))]) == 0)[0][0])].name}")
        return fig, ax
    
def main():
    Ca = atom("Ca.csv", 657.278e-9, 2.6e3, oven_T = 600 + 273.15)
    fig, ax = Ca.plot_spectra(Ca.freq - 0.1e9, Ca.freq + 2.5e9, 1000000, True, True)
    plt.xscale("log")
    plt.yscale("log")
    # fig, ax = Ca.plot_absorp_signal(Ca.freq - 0.1e9, Ca.freq + 2.5e9, 1000000, True, True, False)
    plt.legend()
    plt.show()
    # Sr = atom("Sr.csv", 460.7333e-9, 2.01e8)
    # fig, ax = Sr.plot_spectra(Sr.freq - 0.5e9, Sr.freq + 0.5e9, 1000000, True, True)
    # plt.legend()
    plt.show()
if __name__ == "__main__":
    main()