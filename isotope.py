import matplotlib.pyplot as plt
import scipy.constants as C
import numpy as np

class isotope:
    def __init__(self, name = "", mass = 0, shift = 0, abund = 0, freq = 0):
        self.name = name
        self.mass = mass * C.atomic_mass
        self.shift = shift
        self.abund = abund
        self.freq = freq + shift
        pass
    def __str__(self):
        return f"""
            ========================================================
            {self.name}

            Mass (a.u.):\t{self.mass:.5e}
            Trans. Freq. (Hz):\t{self.freq - self.shift:.5e}
            Iso. Shift (Hz):\t{self.shift:.5e}
            Abundance:\t\t{self.abund * 100:.3}%
            --------------------------------------------------------
            """
    
    def spectra(self, freqs, w0, lw, dshift):
        return self.abund / ((lw / 2) ** 2 + (freqs - w0 + dshift) ** 2)
    def diff_absorp_sig(self, freqs, w0, lw, dshift):
        absMax = (lw / 2) ** 2 + (freqs - w0 + dshift) ** 2
        absMin = (lw / 2) ** 2 + (freqs - w0 - dshift) ** 2
        return (4 * dshift * (w0 - freqs)) * self.abund / (absMax * absMin)
    def doppler(self, w0, vel, theta):
        return w0 * (vel / C.c) * np.sin(theta)
    def average_vel(self, T):
        return np.sqrt(2 * C.k * T / self.mass)
    def saturation(dF, wl):
        return (np.pi * C.h * C.c * dF) / (3 * np.power(wl, 3))
def main():
    # A = 2.6e3
    # Gamma = A / (np.pi * 2)
    # freq = C.c / 657.278e-9
    # Ca = np.array([isotope(r"$^{40}$Ca", 39.962590850, 0, 0.9694, freq), isotope(r"$^{42}$Ca", 41.95861778, 509.5e6, 0.0065, freq), isotope(r"$^{43}$Ca", 42.95876638, 782.2e6, 0.0014, freq), isotope(r"$^{44}$Ca", 43.95548149, 996.2e6, 0.0209, freq), isotope(r"$^{46}$Ca", 45.9536877, 1481.1e6, 4e-5, freq), isotope(r"$^{48}$Ca", 47.952522654, 1922.5e6, 0.0019, freq)])
    # freqs = np.linspace(freq - .25e9, freq + 5e9, 100000, endpoint = True)
    # tot_abs = np.zeros_like(freqs)
    # for iso in Ca:
    #     plt.plot(freqs / 1e9, iso.absorp_sig(freqs, iso.freq, Gamma, iso.doppler(iso.freq, iso.average_vel(900), 0.02609)), ls = "--")    
    #     plt.axvline(iso.freq / 1e9, ls = "--")
    #     print(f"{iso.name} wavelen.: {np.round(1e9 * C.c / (iso.freq - iso.shift), 5)}")
    #     print(f"{iso.name} wavelen. shifted: {np.round(1e9 * C.c / (iso.freq), 5)}")
    #     print()
    #     tot_abs += iso.absorp_sig(freqs, iso.freq, Gamma, iso.doppler(iso.freq, iso.average_vel(900), 0.02609))
    # plt.plot(freqs / 1e9, tot_abs, zorder = 1)
    # plt.show()
    
    # A = 2.01e8
    # Gamma = A / (np.pi * 2)
    # freq = C.c / 460.73330e-9
    # Sr = np.array([isotope(r"$^{84}$Sr", 83.913419, -270.8e6, 0.0056, freq), isotope(r"${86}$Sr", 85.90926073, -124.8e6, 0.0986, freq), isotope(r"$^{87}$Sr", 86.90887750, -68.9e6, 0.07, freq), isotope(r"$^{88}$Sr", 87.900561226, 0, 0.8258, freq)])
    # freqs = np.linspace(freq - 0.5e9, freq + 0.5e9, 100000, endpoint = True)
    # tot_abs = np.zeros_like(freqs)
    # for iso in Sr:
    #     plt.plot(freqs / 1e9, iso.absorp_sig(freqs, iso.freq, Gamma, iso.doppler(iso.freq, iso.average_vel(900), 0.02609)), ls = "--")
    #     plt.axvline(iso.freq / 1e9, ls = "--")
    #     print(f"{iso.name} wavelen.: {np.round(1e9 * C.c / (iso.freq - iso.shift), 5)}")
    #     print(f"{iso.name} wavelen. shifted: {np.round(1e9 * C.c / (iso.freq), 5)}")
    #     print()
    #     tot_abs += iso.absorp_sig(freqs, iso.freq, Gamma, iso.doppler(iso.freq, iso.average_vel(900), 0.02609))
    # plt.plot(freqs / 1e9, tot_abs, zorder = 1)
    # plt.show()
    return

if __name__ == "__main__":
    main()