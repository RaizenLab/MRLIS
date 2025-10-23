import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as con
matplotlib.use('TkAgg')

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
    thetaMax = 0.0195
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
    shift = doppler(freq, vOven, thetaMax)
    print('Doppler Shift with Adjusted velocity: ' + str(shift*1e-6) + 'MHz')
    # absS = np.array(absorptionSignal(frequencies,freq, linewidth, shift))
    # plt.plot(frequencies,absS)
    # plt.show()
    
    ## Frequency Range for Plotting
    minFreq = freq - 0.5e9
    maxFreq = freq + 0.5e9
    frequencies = np.linspace(minFreq, maxFreq, num=1000, endpoint=True)
    
    # Calculate total absorption signal
    total_abs = np.zeros_like(frequencies)
    plt.figure(figsize=(10, 6))

    for isotope in isotopes:
        mass = isotope["mass"]
        mass = isotope["mass"]
        iso_shift = isotope["shift"]
        abundance = isotope["abundance"]
        name = isotope["name"]

        # Resonance frequency for this isotope (Sr-88 freq + shift)
        w0 = freq + iso_shift

        # Calculate Doppler shift for this isotope
        vOven = averageV(ovenTemp, mass)
        doppler_shift = doppler(w0, vOven, thetaMax)

        # Calculate absorption signal
        abs_signal = absorptionSignal(frequencies, w0, linewidth, doppler_shift)
        abs_signal *= abundance  # Weight by abundance
        total_abs += abs_signal

    
    # Plot total absorption signal first (background)
    plt.plot(frequencies / 1e9, total_abs, label="Total Signal", color="black", 
             linewidth=2.5, alpha=0.5, zorder=1)
    
    # Colors for individual isotope plots
    colors = ['blue', 'green', 'red', 'purple']

    # Plot individual isotope signals on top
    for i, isotope in enumerate(isotopes):
        mass = isotope["mass"]
        iso_shift = isotope["shift"]
        abundance = isotope["abundance"]
        name = isotope["name"]
        w0 = freq + iso_shift
        vOven = averageV(ovenTemp, mass)
        doppler_shift = doppler(w0, vOven, thetaMax)
        abs_signal = absorptionSignal(frequencies, w0, linewidth, doppler_shift)
        abs_signal *= abundance
        plt.plot(frequencies / 1e9, abs_signal, label=f"{name} (shift: {iso_shift/1e6:.1f} MHz)", 
                 linestyle='--', color=colors[i], linewidth=1.5, alpha=0.9, zorder=2)
    #  Plot settings
    plt.xlabel("Frequency (GHz)", fontsize=12)
    plt.ylabel("Absorption Signal (arb. units)", fontsize=12)
    plt.title("Absorption Spectra for Strontium Isotopes (Relative to Sr-88)", fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    # plt.grid(True, which="both", linestyle='-', alpha=0.5)
    # plt.minorticks_on()

    
    # New plot for isotopeSpectra
    plt.figure(figsize=(10, 6))
    total_spectra = np.zeros_like(frequencies)

    for isotope in isotopes:
        mass = isotope["mass"]
        iso_shift = isotope["shift"]
        abundance = isotope["abundance"]
        name = isotope["name"]
        w0 = freq + iso_shift
        vOven = averageV(ovenTemp, mass)
        doppler_shift = doppler(w0, vOven, thetaMax)
        spectra = isotopeSpectra(frequencies, w0, linewidth, doppler_shift)
        spectra *= abundance  # Weight by abundance
        total_spectra += spectra

    # Plot total isotope spectra (background)
    plt.plot(frequencies / 1e9, total_spectra, label="Total Isotope Spectra", color="black", 
             linewidth=2.5, alpha=0.5, zorder=1)

    # Plot individual isotope spectra
    for i, isotope in enumerate(isotopes):
        mass = isotope["mass"]
        iso_shift = isotope["shift"]
        abundance = isotope["abundance"]
        name = isotope["name"]
        w0 = freq + iso_shift
        vOven = averageV(ovenTemp, mass)
        doppler_shift = doppler(w0, vOven, thetaMax)
        spectra = isotopeSpectra(frequencies, w0, linewidth, doppler_shift)
        spectra *= abundance
        plt.plot(frequencies / 1e9, spectra, label=f"{name} (shift: {iso_shift/1e6:.1f} MHz)", 
                 linestyle='--', color=colors[i], linewidth=1.5, alpha=0.9, zorder=2)
        
    plt.show()

if __name__ == "__main__":
    main()