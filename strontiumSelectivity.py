import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as con

# Set the matplotlib backend to 'TkAgg' for compatibility
matplotlib.use('TkAgg')

# Code written by Henry R. Chance
# This code is intended to calculate the slectivity of strontium isotopes in MRLIS

def calculate_excited_abundances(target_iso, all_isotopes, linewidth):
    """
    Calculates the new fractional abundance (purity) of *all* isotopes 
    in the excited population when the laser is tuned to a single target isotope.

    Args:
        target_iso (dict): The isotope dictionary for the target.
        all_isotopes (list): The complete list of isotope dictionaries.
        linewidth (float): The natural linewidth (FWHM, Gamma) of the transition.

    Returns:
        dict: A dictionary of {isotope_name: new_abundance} showing the
              percentage of each isotope in the excited population.
    """
    
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
            # Using your formula: S ≈ (2*delta_nu / Gamma)^2
            # Note: The more accurate formula is S = 1 + (2*delta_nu / Gamma)^2
            # Using your formula as requested:
            selectivity_over_iso = (2 * delShift / linewidth)**2
            
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
            new_abundances[name] = contribution / total_excited_population
            
    return new_abundances

def isotopeSpectra(w, w0, dw, dShift):
    """
    Calculates the value of a Lorentzian lineshape at a given frequency 'w'.
    This function is not normalized (peak height is not 1).
    Args:
        w (float or np.array): The frequency (or frequencies) to evaluate at.
        w0 (float): The center resonance frequency of the transition.
        dw (float): The natural linewidth (FWHM, or Gamma) of the transition.
        dShift (float): An additional frequency shift (e.g., Doppler shift), 
                        though it's set to 0 when called in main().
                        
    Returns:
        float or np.array: The absorption profile value(s).
    """
    # The Lorentzian formula is proportional to 1 / ( (Gamma/2)^2 + (detuning)^2 )
    # Here, detuning = (w - (w0 - dShift))
    absPlus = (dw / 2)**2 + (w - w0 + dShift)**2
    # The function returns (1 / (Gamma/2)^2) * [ (Gamma/2)^2 / ( (Gamma/2)^2 + (detuning)^2 ) ]
    # This is a scaled Lorentzian.
    res = 1 / absPlus
    return res

def calculate_selectivity_and_purity(isotopes, linewidth):
    """
    Calculates and prints the selectivity and purity for each isotope.
    Args:
        isotopes (list): List of isotope dictionaries.
        linewidth (float): Natural linewidth (FWHM, Gamma) of the transition.
    """
    print("\n--- Isotope Selectivity and Purity Analysis ---")

    # Loop through each isotope, treating it as the target
    for target_iso in isotopes:
        # --- Selectivity Calculation ---
        # Laser is tuned to the target isotope's resonance frequency
        
        # 'min' will store the smallest frequency shift found so far (nearest neighbor)
        min_shift_found = np.inf  # Initialize with infinity
        # 'denom' is the denominator for the purity calculation.
        # It starts with the target's own contribution: f_A
        denom = target_iso['abundance']
        
        # 'selectivity' will store the selectivity vs. the nearest neighbor
        selectivity = 0.0

        # Loop through all isotopes to find the competitors
        for i, iso in enumerate(isotopes):
            # Skip if this is the target isotope
            if iso['name'] == target_iso['name']:
                continue

            # delShift is the isotope shift between the target and the competitor
            delShift = np.abs(target_iso['shift'] - iso['shift'])
            
            # The Selectivity is given by S = 1 + (2*delta_nu / Gamma)^2.
            select = (2 * delShift / linewidth)**2
            
            # This logic finds the selectivity for the nearest neighbor
            if not min_shift_found or delShift < min_shift_found:
                min_shift_found = delShift
                selectivity = select
            
            # Add the competitor's leakage term (f_i / S_i) to the denominator
            denom += iso['abundance'] / select
        
        # Calculate Purity with complete denominator
        purity = target_iso['abundance'] / denom

        # Print the combined string for this target isotope
        print(f"{target_iso['name']}: Selectivity (vs nearest neighbor): {selectivity:.2f},  Purity (vs all others): {purity * 100:.2f}%")

def main():
    # Experiment Parameters
    thetaMax = 0.0195  # Collimation angle
    waveL = 460.73330e-9  # Wavelength for Sr-88 transition
    freq = con.c / waveL  # Convert wavelength to frequency (Hz)
    ovenTemp = 530 + 273.15  # Oven temperature in Kelvin
    mSr = 1.4549642e-25  # Mass of Sr-88
    
    # Lifetime of the 5s5p 1P1 state is ~5.2 ns
    # Einstein A coefficient A = 1/tau
    einsteinA = 2.01e8  # A coefficient (from 1/4.97e-9 s)
    # Natural Linewidth (FWHM) Gamma = A / (2*pi)
    linewidth = (einsteinA) / (2 * con.pi) # This is 31.99 MHz

    # Isotope Parameters (from NIST)
    # 'shift' is the frequency shift relative to Sr-88 (in Hz)
    isotopes = [
        {"mass": 83.913419 * con.atomic_mass, "shift": -270.8e6, "abundance": 0.0056, "name": "Sr-84"},
        {"mass": 85.90926073 * con.atomic_mass, "shift": -124.8e6, "abundance": 0.0986, "name": "Sr-86"},
        {"mass": 86.90887750 * con.atomic_mass, "shift": -68.9e6, "abundance": 0.0700, "name": "Sr-87"},
        {"mass": 87.90561226 * con.atomic_mass, "shift": 0.0, "abundance": 0.8258, "name": "Sr-88"},
    ]

    # Calculate Selectivities
    calculate_selectivity_and_purity(isotopes, linewidth)

    # Calculate the New Abundances When each Isotope is targeted
    print("\n--- New Abundances in Excited Population ---")
    for target in isotopes:
        print(f"  Targeting {target['name']}:")
        abundances = calculate_excited_abundances(target, isotopes, linewidth)
        for name, abundance in abundances.items():
            print(f"    {name}: {abundance * 100:.4f}%")


    ## Frequency Range and Colors for Plotting
    # Set up a frequency range for plotting, centered around Sr-88
    minFreq = freq - 0.5e9
    maxFreq = freq + 0.5e9
    frequencies = np.linspace(minFreq, maxFreq, num=1000, endpoint=True)
    colors = ['blue', 'green', 'red', 'purple']

    # --- Plotting Section ---
    plt.figure(figsize=(10, 6))
    plt.title("Absorption Profiles for Strontium Isotopes", fontsize=14)
    # X-axis is frequency shift from Sr-88, in GHz
    plt.xlabel("Frequency Shift from Sr-88 (GHz)", fontsize=12)
    plt.ylabel("Normalized Absorption (arb. units)", fontsize=12)

    # Plot each isotope's profile
    for i, isotope in enumerate(isotopes):
        iso_shift = isotope["shift"]
        name = isotope["name"]

        # w0 is the absolute center frequency for this isotope
        w0 = freq + iso_shift

        # Calculate the Lorentzian profile
        # We set the 'dShift' argument to 0 because we are already plotting
        # vs. absolute frequency and have set w0 correctly.
        spectra = isotopeSpectra(frequencies, w0, linewidth, 0)

        # Scale the height of the plot by the isotope's natural abundance
        spectra *= isotope["abundance"]
        
        # Plot X-axis as shift from Sr-88 (freq) in GHz
        # Plot Y-axis as the abundance-scaled absorption
        plt.plot((frequencies - freq) / 1e9, spectra, label=f"{name} ({isotope['abundance']*100:.1f}%) [{iso_shift/1e6:.1f} MHz]",
                 color=colors[i], linewidth=2)
        # Add a vertical dashed line at the center of each isotope's peak
        plt.axvline(x=(w0 - freq) / 1e9, color=colors[i], linestyle='--', linewidth=1, alpha=0.8)

    # Zoom in on the x-axis to see the peaks clearly
    plt.xlim(-0.4, 0.25)
    plt.ylabel("Relative Absorption (Abundance * Profile)", fontsize=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6) # Add a light grid
    plt.tight_layout() # Adjust plot to prevent label cutoff
    plt.show()

if __name__ == "__main__":
    main()