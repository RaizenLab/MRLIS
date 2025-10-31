import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as con

def calculate_excited_abundances(target_iso, all_isotopes, linewidth, intensity, I_sat,):
    # This calcualtes the power broadened selectivity and resulting purity of an isotope selective transition
        # This dictionary will store the numerator of the purity formula for each isotope.
    # Create new dictionary to store new abundances saving original structure
    new_iso = []

    printPlz = True  # Print results to console
    # For target A: f_A
    # For competitor B: f_B / S_B .where S_B is the selectivity of A over B.
    relative_excited_contributions = {}
    
    # This will be the sum of all relative_excited_contributions, (f_A + f_B/S_B + ...), which is the denominator in the purity formula.
    total_excited_population = 0.0

    # Get the resonance frequency of the target isotope (our laser frequency)
    laser_frequency_shift = target_iso['shift']
    if linewidth > 0:
        for iso in all_isotopes:
            # Get the frequency shift between the laser and this isotope
            delShift = np.abs(laser_frequency_shift - iso['shift'])

            # Calculate the selectivity of the target (laser) over this isotope
            # S_i = S_target / S_iso
            selectivity_over_iso = 1 + ((2 * delShift) / (linewidth*np.sqrt(1+intensity/I_sat)))**2

            # Calculate this isotope's relative contribution to the excited state
            # contribution = f_i / S_i
            contribution = iso['abundance'] / selectivity_over_iso
            
            relative_excited_contributions[iso['name']] = contribution
            total_excited_population += contribution

        for iso in all_isotopes:
            new_iso_data = iso.copy()  # Create a copy to avoid modifying the original list
            new_iso_data['abundance'] = relative_excited_contributions[iso['name']]/total_excited_population
            new_iso.append(new_iso_data)
            if printPlz:
                print(f"    {iso['name']}: {new_iso_data['abundance'] * 100:.4f}%")

    return new_iso, total_excited_population

def calculate_unexcited_abundances(target_iso, all_isotopes, linewidth, intensity, I_sat):
    # This calcualtes the power broadened selectivity and resulting purity of an isotope selective transition
    
    # This dictionary will store the numerator of the purity formula for each isotope.
    # For target A: f_A
    # For competitor B: f_B / S_B .where S_B is the selectivity of A over B.
    relative_excited_contributions = {}
    
    # This will be the sum of all relative_excited_contributions, (f_A + f_B/S_B + ...), which is the denominator in the purity formula.
    total_excited_population = 0.0

    # Get the resonance frequency of the target isotope (our laser frequency)
    laser_frequency_shift = target_iso['shift']

    for iso in all_isotopes:
        # Get the frequency shift between the laser and this isotope
        delShift = np.abs(laser_frequency_shift - iso['shift'])

        # Calculate the selectivity of the target (laser) over this isotope
        # S_i = S_target / S_iso
        selectivity_over_iso = 1.0 # Default for the target itself

        if iso['name'] != target_iso['name']:
            # Using the power broadened selectivity formula
            selectivity_over_iso = 1 + ((2 * delShift) / (linewidth*np.sqrt(1+intensity/I_sat)))**2
            
            # Prevent division by zero if two isotopes have identical shifts
            if selectivity_over_iso == 0:
                # This implies the laser is also resonant with this isotope
                selectivity_over_iso = 1.0 

        # Calculate this isotope's relative contribution to the excited state
        # contribution = f_i / S_i
        contribution = iso['abundance'] / selectivity_over_iso
        
        relative_excited_contributions[iso['name']] = contribution
        total_excited_population += contribution

    # --- Normalize to get final abundances (purities) ---
    new_abundances = {}
    if total_excited_population == 0:
        # Handle case where no isotopes are excited (e.g., linewidth is 0)
        for iso in all_isotopes:
            new_abundances[iso['name']] = 0.0
    else:
        # Purity_i = (f_i / S_i) / (f_A + f_B/S_B + f_C/S_C + ...)
        for name, contribution in relative_excited_contributions.items():
            new_abundances[name] = 1 - contribution / total_excited_population

    return new_abundances, total_excited_population


def main():
    # Experiment Parameters
    thetaMax = 0.0195  # Collimation angle
    ovenTemp = 530 + 273.15  # Oven temperature in Kelvin
    mSr = 1.4549642e-25  # Mass of Sr-88
    powerReduce = 1/100  # Factor to reduce laser power by (to account for losses)
    selection_intensity = powerReduce*5658.84 # Intensity of laser W/m^2 (1W with 1.5cm spot size)
    Nstart = 1e14*24*3600  # Initial number of atoms in the interaction region after 1 day of operation
    print("\n--- Sr Isotope Selectivity Calculations ---")
    print("\n--- Initial Parameters ---")
    print((f"    Initial number of atoms in interaction region after 1 day: {Nstart:.2e} atoms"))

    # Isotope Parameters (from NIST)
    # 'shift' is the frequency shift relative to Sr-88 (in Hz)
    isotopes461 = [
        {"mass": 83.913419 * con.atomic_mass, "shift": -270.8e6, "abundance": 0.0056, "name": "Sr-84"},
        {"mass": 85.90926073 * con.atomic_mass, "shift": -124.8e6, "abundance": 0.0986, "name": "Sr-86"},
        {"mass": 86.90887750 * con.atomic_mass, "shift": -68.9e6, "abundance": 0.0700, "name": "Sr-87"},
        {"mass": 87.90561226 * con.atomic_mass, "shift": 0.0, "abundance": 0.8258, "name": "Sr-88"},
    ]
    waveL_461 = 460.73330e-9  # Wavelength for Sr-88 transition
    freq461 = con.c / waveL_461  # Convert wavelength to frequency (Hz)
    # Lifetime of the 5s5p 1P1 state is ~5.2 ns
    # Einstein A coefficient A = 1/tau
    einsteinA_461 = 2.01e8  # A coefficient (from 1/4.97e-9 s)
    # Natural Linewidth (FWHM) Gamma = A / (2*pi)
    linewidth_461 = (einsteinA_461) / (2 * con.pi) # This is 31.99 MHz
    # This is the saturation intensity of the transition
    Isat_461 = (con.pi*con.h*con.c*linewidth_461/(3*waveL_461**3))
    print(f"    461nm Isat: {Isat_461:.2f} W/m^2")


    # Isotope shifts for the 689 nm line (in Hz)
    isotopes689 = [
        {"mass": 83.913419 * con.atomic_mass, "shift": -351.49e6, "abundance": 0.0056, "name": "Sr-84"},
        {"mass": 85.90926073 * con.atomic_mass, "shift": -163.8174e6, "abundance": 0.0986, "name": "Sr-86"},
        {"mass": 86.90887750 * con.atomic_mass, "shift": 221.71e6, "abundance": 0.0700, "name": "Sr-87"},
        {"mass": 87.90561226 * con.atomic_mass, "shift": 0.0, "abundance": 0.8258, "name": "Sr-88"},
    ]
    waveL_689 = 689.41434e-9  # Wavelength for Sr-88 689 nm transition
    freq689 = con.c / waveL_689  # Convert wavelength to frequency (Hz)
    einsteinA_689 = 4.7e4  # A coefficient for 689 nm transition
    linewidth_689 = einsteinA_689 / (2 * con.pi)  # Natural Linewidth for 689 nm transition
    Isat_689 = (con.pi*con.h*con.c*linewidth_689/(3*waveL_689**3))
    print(f"    689nm Isat: {Isat_689:.2f} W/m^2")
    

    # Isotope shifts for 655 nm line (after 461nm excitation)
    isotopes655 = [
        {"mass": 83.913419 * con.atomic_mass, "shift": -785.2e6, "abundance": 0.0056, "name": "Sr-84"},
        {"mass": 85.90926073 * con.atomic_mass, "shift": -350.6e6, "abundance": 0.0986, "name": "Sr-86"},
        {"mass": 86.90887750 * con.atomic_mass, "shift": -186.1e6, "abundance": 0.0700, "name": "Sr-87"},
        {"mass": 87.90561226 * con.atomic_mass, "shift": 0.0, "abundance": 0.8258, "name": "Sr-88"},
    ]
    waveL_655 = 655.873e-9  # Wavelength for Sr-88 655 nm transition
    freq655 = con.c / waveL_655  # Convert wavelength to frequency (Hz
    einsteinA_655 = 8.9e7  # A coefficient for 655 nm transition
    linewidth_655 = einsteinA_655 / (2 * con.pi)  # Natural Linewidth for 655 nm transition
    Isat_655 = (con.pi*con.h*con.c*linewidth_655/(3*waveL_655**3))
    print(f"    655nm Isat: {Isat_655:.2f} W/m^2")



    ## Calculate spectra after two passes through 461 pathway
    print("\n--- Case 1: Two Passes through 461nm Ionization Pathway ---")
    goalIsotope = isotopes461[0]  # Target Sr-84 for enrichment
    print(f"\n Step 1: Abundances after First Pass (Targeting {goalIsotope['name']})")
    firstpass_iso461, firstpass461_exitedFraction = calculate_excited_abundances(goalIsotope, isotopes461, linewidth_461, selection_intensity, Isat_461)
    NfirstPass461 = firstpass461_exitedFraction*Nstart*firstpass_iso461[0]['abundance']
    print(f"    Number of Excited Sr84 after First Pass: {NfirstPass461:.2e} atoms")
    

    print(f"\n Step 2: Abundances after Second Pass (Targeting {goalIsotope['name']})")
    secondpass_iso461, secondpass461_exitedFraction = calculate_excited_abundances(goalIsotope, firstpass_iso461, linewidth_461, selection_intensity, Isat_461)
    NsecondPass461 = secondpass461_exitedFraction*NfirstPass461*secondpass_iso461[0]['abundance']
    print(f"    Number of Excited Sr84 after Second Pass: {NsecondPass461:.2e} atoms")
    

if __name__ == "__main__":
    main()