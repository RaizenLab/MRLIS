import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as con
matplotlib.use('TkAgg')
from scipy import integrate

# Code written by Henry R. Chance
# This code is intended to calculate the absorption spectra of strontium isotopes

def averageV(T,m):
    #vav = np.sqrt((8*con.k*T)/(m*con.pi))
    vav = np.sqrt((2*con.k*T)/(m))
    return vav

def doppler(w0,v,thet):
    dFreq  = w0*(v/con.c)*np.sin(thet)
    return dFreq

def isotopeSpectra(w,w0,dw,dShift):
    absPlus = (dw/2)**2 + (w-w0+dShift)**2
    res = 1/absPlus
    return res

def absorptionSignal(w,w0,dw,dShift):
    absPlus = (dw/2)**2 + (w-w0+dShift)**2
    absMin = (dw/2)**2 + (w-w0-dShift)**2
    res = (4*dShift*(w0-w))/(absPlus*absMin)
    return res



def main():
    # Experiment Parameters
    # thetaMax = 0.0195
    atomBeamDivergence = 0.02609 # Measured atomic beam divergence (radians)
    thetaLaser = 0.0351 # Angle of the laser beams with respect to the normal (35.1 mrad)
    waveL = 460.73330e-9 # From https://physics.nist.gov/PhysRefData/ASD/lines_form.html (Sr-88, 5s^2 1S0 -> 5s5p 1P1)
    freq = con.c/waveL
    ovenTemp = 530+273.15
    mSr = 1.4549642e-25
    einsteinA = 2.01e8
    linewidth = (einsteinA)/(2*con.pi)

    # Isotope Parameters (from NIST)
    isotopes = [
        {"mass": 83.913419 * con.atomic_mass, "shift": -270.8e6, "abundance": 0.0056, "name": "Sr-84"},
        {"mass": 85.90926073 * con.atomic_mass, "shift": -124.8e6, "abundance": 0.0986, "name": "Sr-86"},
        {"mass": 86.90887750 * con.atomic_mass, "shift": -68.9e6, "abundance": 0.0700, "name": "Sr-87"},
        {"mass": 87.90561226 * con.atomic_mass, "shift": 0.0, "abundance": 0.8258, "name": "Sr-88"},
    ]

    ## Use for single isotope
    vOven = averageV(ovenTemp, mSr)
    shift = doppler(freq, vOven, atomBeamDivergence)
    print('Doppler Shift with Adjusted velocity: ' + str(shift*1e-6) + 'MHz')

    # Colors for individual isotope plots
    colors = ['blue', 'green', 'red', 'purple']

    ## Frequency Range for Plotting
    minFreq = freq - 0.5e9
    maxFreq = freq + 0.5e9
    frequencies = np.linspace(minFreq, maxFreq, num=1000, endpoint=True)
    
    total_abs = np.zeros_like(frequencies)
    plt.figure(figsize=(6.5, 4.875))

    for i, isotope in enumerate(isotopes):
        mass = isotope["mass"]
        iso_shift = isotope["shift"]
        abundance = isotope["abundance"]
        name = isotope["name"]
        w0 = freq + iso_shift
        vOven = averageV(ovenTemp, mass)
        f = lambda alpha, v: ((1/((linewidth/2)**2+(frequencies-w0+w0*(v/con.c)*np.sin(thetaLaser-alpha))**2))-(1/((linewidth/2)**2+(frequencies-w0+w0*(v/con.c)*np.sin(-thetaLaser-alpha))**2)))*(1/atomBeamDivergence)*((2*v**3)/vOven**4)*np.exp(-v**2/vOven**2)
        integral = integrate.dblquad(f, -atomBeamDivergence, atomBeamDivergence, 0, np.inf)
        total_abs += abundance*integral[0]
        plt.axvline(x=(w0) / 1e9, color=colors[i], linestyle='--', linewidth=1, label=f"{name}", alpha=0.8)
    
    plt.plot(frequencies / 1e9, total_abs, color="black", linewidth=2.5, alpha=0.5, zorder=1)
    plt.xlabel("Frequency (GHz)", fontsize=14)
    plt.ylabel("Absorption Signal (arb. units)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.title("Absorption Spectra for Strontium Isotopes (Relative to Sr-88) [More Accurate]", fontsize=16)
    plt.legend(loc="upper right", fontsize=12)
    plt.show()






    plt.figure(figsize=(6.5, 4.875))
    plt.title("Absorption Signal Integration Test")
    test_w0 = freq
    test_linewidth = linewidth
    test_doppler_shift = shift
    abs_signal_test = absorptionSig(frequencies, test_w0, test_linewidth, test_doppler_shift)
    plt.plot(frequencies / 1e9, abs_signal_test, label="Absorption Signal", color="blue")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Absorption Signal (arb. units)")
    plt.legend()
    plt.show()
    # Calculate total absorption signal
    total_abs = np.zeros_like(frequencies)
    plt.figure(figsize=(1.15*6.5, 1.15*4.875))

    for isotope in isotopes:
        mass = isotope["mass"]
        iso_shift = isotope["shift"]
        abundance = isotope["abundance"]
        name = isotope["name"]

        # Resonance frequency for this isotope (Sr-88 freq + shift)
        w0 = freq + iso_shift

        # Calculate Doppler shift for this isotope
        # The absorptionSignal function uses this magnitude to create both 
        # positive and negative shifted components (+/- doppler_shift)
        vOven = averageV(ovenTemp, mass)
        doppler_shift = doppler(w0, vOven, thetaMax + thetaLaser)

        # Calculate absorption signal
        abs_signal = absorptionSignal(frequencies, w0, linewidth, doppler_shift)
        abs_signal *= abundance  # Weight by abundance
        total_abs += abs_signal

    
    # Plot total absorption signal first (background), label="Total Signal"
    plt.plot(frequencies / 1e9, total_abs, color="black", 
             linewidth=2.5, alpha=0.5, zorder=1)
    
    # Plot y=0
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
    

    # Plot individual isotope signals on top
    for i, isotope in enumerate(isotopes):
        mass = isotope["mass"]
        iso_shift = isotope["shift"]
        abundance = isotope["abundance"]
        name = isotope["name"]
        w0 = freq + iso_shift
        vOven = averageV(ovenTemp, mass)
        doppler_shift = doppler(w0, vOven, atomBeamDivergence + thetaLaser)
        abs_signal = absorptionSignal(frequencies, w0, linewidth, doppler_shift)
        abs_signal *= abundance
        # plt.plot(frequencies / 1e9, abs_signal, label=f"{name}", 
        #          linestyle='--', color=colors[i], linewidth=1.5, alpha=0.9, zorder=2)
        plt.axvline(x=(w0) / 1e9, color=colors[i], linestyle='--', linewidth=1, label=f"{name}", alpha=0.8)
        #  (shift: {iso_shift/1e6:.1f} MHz)
    #  Plot settings
    plt.xlabel("Frequency (GHz)", fontsize=14)
    plt.ylabel("Absorption Signal (arb. units)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.title("Absorption Spectra for Strontium Isotopes (Relative to Sr-88)", fontsize=16)
    plt.legend(loc="upper right", fontsize=12)
    # plt.grid(True, which="both", linestyle='-', alpha=0.5)
    # plt.minorticks_on()

    
    # New plot for isotopeSpectra
    # plt.figure(figsize=(6.5, 4.875))
    total_spectra = np.zeros_like(frequencies)

    for isotope in isotopes:
        mass = isotope["mass"]
        iso_shift = isotope["shift"]
        abundance = isotope["abundance"]
        name = isotope["name"]
        w0 = freq + iso_shift
        vOven = averageV(ovenTemp, mass)
        doppler_shift = doppler(w0, vOven, atomBeamDivergence + thetaLaser)
        spectra = isotopeSpectra(frequencies, w0, linewidth, doppler_shift)
        spectra *= abundance  # Weight by abundance
        total_spectra += spectra

    # # Plot total isotope spectra (background)
    # plt.plot(frequencies / 1e9, total_spectra, label="Total Isotope Spectra", color="black", 
    #          linewidth=2.5, alpha=0.5, zorder=1)

    # # Plot individual isotope spectra
    # for i, isotope in enumerate(isotopes):
    #     mass = isotope["mass"]
    #     iso_shift = isotope["shift"]
    #     abundance = isotope["abundance"]
    #     name = isotope["name"]
    #     w0 = freq + iso_shift
    #     vOven = averageV(ovenTemp, mass)
    #     doppler_shift = doppler(w0, vOven, thetaMax)
    #     spectra = isotopeSpectra(frequencies, w0, linewidth, doppler_shift)
    #     spectra *= abundance
    #     plt.plot(frequencies / 1e9, spectra, label=f"{name} (shift: {iso_shift/1e6:.1f} MHz)", 
    #              linestyle='--', color=colors[i], linewidth=1.5, alpha=0.9, zorder=2)
        
    plt.show()

if __name__ == "__main__":
    main()