import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as con

isotopes461 = [
    {"mass": 83.913419 * con.atomic_mass, "shift": -270.8e6, "abundance": 0.0056, "name": "Sr-84"},
    {"mass": 85.90926073 * con.atomic_mass, "shift": -124.8e6, "abundance": 0.0986, "name": "Sr-86"},
    {"mass": 86.90887750 * con.atomic_mass, "shift": -68.9e6, "abundance": 0.0700, "name": "Sr-87"},
    {"mass": 87.90561226 * con.atomic_mass, "shift": 0.0, "abundance": 0.8258, "name": "Sr-88"},
]

isotopes689 = [
    {"mass": 83.913419 * con.atomic_mass, "shift": -351.49e6, "abundance": 0.0056, "name": "Sr-84"},
    {"mass": 85.90926073 * con.atomic_mass, "shift": -163.8174e6, "abundance": 0.0986, "name": "Sr-86"},
    {"mass": 86.90887750 * con.atomic_mass, "shift": 221.71e6, "abundance": 0.0700, "name": "Sr-87"},
    {"mass": 87.90561226 * con.atomic_mass, "shift": 0.0, "abundance": 0.8258, "name": "Sr-88"},
]

isotopes655 = [
    {"mass": 83.913419 * con.atomic_mass, "shift": -785.2e6, "abundance": 0.0056, "name": "Sr-84"},
    {"mass": 85.90926073 * con.atomic_mass, "shift": -350.6e6, "abundance": 0.0986, "name": "Sr-86"},
    {"mass": 86.90887750 * con.atomic_mass, "shift": -186.1e6, "abundance": 0.0700, "name": "Sr-87"},
    {"mass": 87.90561226 * con.atomic_mass, "shift": 0.0, "abundance": 0.8258, "name": "Sr-88"},
]

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

def calculate_unexcited_abundances(target_iso, all_isotopes, 
                                     linewidth, intensity, I_sat):
    """
    Calculates the new (depleted) isotopic abundances in the 
    unexcited (neutral) state, assuming full saturation 
    of the target transition (P_target = 1).
    
    Args:
        target_iso (dict): The target isotope.
        all_isotopes (list): List of all isotopes.
        linewidth (float): The natural (FWHM) linewidth of the transition.
        intensity (float): The intensity of the selective laser.
        I_sat (float): The saturation intensity of the transition.
    """
    
    new_iso_list = []
    relative_unexcited_contributions = {}
    total_excited_fraction = 0.0  # This is your 'total_excited_population' (T_rel)
    
    laser_frequency_shift = target_iso['shift']
    
    if linewidth <= 0:
        print("Error: Linewidth must be > 0.")
        return all_isotopes, total_excited_fraction

    # Calculate the power-broadened linewidth
    power_broadened_width = linewidth * np.sqrt(1.0 + intensity / I_sat)

    # --- This loop calculates both the numerators and the denominator ---
    for iso in all_isotopes:
        detuning = np.abs(laser_frequency_shift - iso['shift'])
        
        # S_i: Selectivity of the target over this isotope
        selectivity_over_iso = 1.0 + ((2.0 * detuning) / power_broadened_width)**2

        # P_i: Excitation probability for this isotope (assuming P_target = 1)
        P_i = 1.0 / selectivity_over_iso
        
        # f_i * P_i: Contribution to the total *excited* fraction
        total_excited_fraction += iso['abundance'] * P_i

        # f_i * (1 - P_i): This isotope's contribution to the *unexcited* pile
        contribution = iso['abundance'] * (1.0 - P_i)
        
        relative_unexcited_contributions[iso['name']] = contribution

    # This is the total fraction of the sample left unexcited.
    # It is the denominator for our new abundance calculation.
    # Note: total_unexcited_fraction = 1.0 - total_excited_fraction
    total_unexcited_fraction = sum(relative_unexcited_contributions.values())

    # --- Now, normalize to get the new abundances ---
    if total_unexcited_fraction == 0:
        # This is a hypothetical case where 100% of all atoms 
        # (not just the target) were excited.
        print("Warning: Total unexcited population is zero (100% excitation).")
        for iso in all_isotopes:
            new_data = iso.copy()
            new_data['abundance'] = 0.0
            new_iso_list.append(new_data)
        return new_iso_list

    for iso in all_isotopes:
        new_iso_data = iso.copy()
        
        # A_i = (Unexcited Contribution_i) / (Total Unexcited Fraction)
        new_abundance = relative_unexcited_contributions[iso['name']] / total_unexcited_fraction
        new_iso_data['abundance'] = new_abundance
        new_iso_list.append(new_iso_data)
        
        print(f"    {iso['name']}: {new_abundance * 100:.4f}%")
        
    return new_iso_list, total_unexcited_fraction

def averageV(T,m):
    #vav = np.sqrt((8*con.k*T)/(m*con.pi))
    vav = np.sqrt((8*con.k*T)/(m))
    return vav

def main():
    global isotopes461, isotopes689, isotopes655
    # Experiment Parameters
    thetaMax = 0.0195  # Collimation angle
    ovenTemp = 530 + 273.15  # Oven temperature in Kelvin
    mSr = 1.4549642e-25  # Mass of Sr-88
    vOven = averageV(ovenTemp, mSr)
    powerReduce = 1/100  # Factor to reduce laser power by (to account for losses)
    selection_intensity = powerReduce*5658.84 # Intensity of laser W/m^2 (1W with 1.5cm spot size)
    Nstart = 1e14*24*3600  # Initial number of atoms in the interaction region after 1 day of operation
    NstartSr84 = Nstart*isotopes461[0]['abundance']
    print("\n---- Sr Isotope Selectivity Calculations ----")
    print("\n--- Initial Parameters ---\n")
    print((f"    Initial number of atoms in interaction region after 1 day: {Nstart:.2e} atoms"))
    print(f"    Average Atom Velocity from Oven: {vOven:.1f} m/s")

    # Isotope Parameters (from NIST)
    # 'shift' is the frequency shift relative to Sr-88 (in Hz)
    waveL_461 = 460.73330e-9  # Wavelength for Sr-88 transition
    freq461 = con.c / waveL_461  # Convert wavelength to frequency (Hz)
    # Lifetime of the 5s5p 1P1 state is ~5.2 ns
    # Einstein A coefficient A = 1/tau
    einsteinA_461 = 2.01e8  # A coefficient (from 1/4.97e-9 s)
    # Natural Linewidth (FWHM) Gamma = A / (2*pi)
    linewidth_461 = (einsteinA_461) / (2 * con.pi) # This is 31.99 MHz
    # This is the saturation intensity of the transition
    Isat_461 = (con.hbar*linewidth_461*(2*con.pi*freq461)**3)/(12*con.pi*con.c**2)
    selection_intensity461 = Isat_461
    satTime461 = 1.0/(einsteinA_461*(1+selection_intensity461/Isat_461))
    satDistance461 = vOven*satTime461
    depletionDistance461 = vOven/einsteinA_461
    print("\n    461nm Laser Parameters:")
    print(f"    461nm Isat: {Isat_461:.2e} W/m^2, 1.5cm Spot Size Psat: {1000*Isat_461*con.pi*(0.015/2)**2:.2e} mW")
    print(f"    Time to Saturate: {satTime461:.2e} s")
    print(f"    Distance to Saturate: {satDistance461:.2e} m, Distance to Depletion: {depletionDistance461:.2e} m")


    # Isotope shifts for the 689 nm line (in Hz)
    waveL_689 = 689.41434e-9  # Wavelength for Sr-88 689 nm transition
    freq689 = con.c / waveL_689  # Convert wavelength to frequency (Hz)
    einsteinA_689 = 4.69e4  # A coefficient for 689 nm transition
    linewidth_689 = einsteinA_689 / (2 * con.pi)  # Natural Linewidth for 689 nm transition
    Isat_689 = (con.hbar*linewidth_689*(2*con.pi*freq689)**3)/(12*con.pi*con.c**2)  # Saturation intensity for 689 nm transition
    selection_intensity689 = Isat_689
    satTime689 = 1.0/(einsteinA_689*(1+selection_intensity689/Isat_689))
    satDistance689 = vOven*satTime689
    depletionDistance689 = vOven/einsteinA_689
    print("\n    689nm Laser Parameters:")
    print(f"    689nm Isat: {Isat_689:.2e} W/m^2, 1.5cm Spot Size Psat: {1000*Isat_689*con.pi*(0.015/2)**2:.2e} mW")
    print(f"    Time to Saturate: {satTime689:.2e} s")
    print(f"    Distance to Saturate: {satDistance689:.2e} m, Distance to Depletion: {depletionDistance689:.2e} m")

    # Laser Parameters for 688nm excitation
    waveL_688 = 687.83134e-9  # Wavelength for Sr-88 688 nm transition
    freq688 = con.c / waveL_688  # Convert wavelength to frequency (Hz)
    einsteinA_688 = 2.7e7  # A coefficient for 688 nm transition
    linewidth_688 = einsteinA_688 / (2 * con.pi)  # Natural Linewidth for 688 nm transition
    Isat_688 = (con.hbar*linewidth_688*(2*con.pi*freq688)**3)/(12*con.pi*con.c**2)
    selection_intensity688 = Isat_688
    satTime688 = 1.0/(einsteinA_688*(1+selection_intensity688/Isat_688))
    satDistance688 = vOven*satTime688
    depletionDistance688 = vOven/einsteinA_688
    print("\n    688nm Laser Parameters:")
    print(f"    688nm Isat: {Isat_688:.2e} W/m^2, 1.5cm Spot Size Psat: {1000*Isat_688*con.pi*(0.015/2)**2:.2e} mW")
    print(f"    Time to Saturate: {satTime688:.2e} s")
    print(f"    Distance to Saturate: {satDistance688:.2e} m, Distance to Depletion: {depletionDistance688:.2e} m")

    # Isotope shifts for 655 nm line (after 461nm excitation)
    waveL_655 = 655.873e-9  # Wavelength for Sr-88 655 nm transition
    freq655 = con.c / waveL_655  # Convert wavelength to frequency (Hz)
    einsteinA_655 = 8.9e7  # A coefficient for 655 nm transition
    linewidth_655 = einsteinA_655 / (2 * con.pi)  # Natural Linewidth for 655 nm transition
    Isat_655 = (con.pi*con.h*con.c*linewidth_655/(3*waveL_655**3))
    selection_intensity655 = 2*Isat_655
    satTime655 = 1.0/(einsteinA_655*(1+selection_intensity655/Isat_655))
    satDistance655 = vOven*satTime655
    depletionDistance655 = vOven/einsteinA_655
    print("\n    655nm Laser Parameters:")
    print(f"    655nm Isat: {Isat_655:.2e} W/m^2")
    print(f"    Time to Saturate: {satTime655:.2e} s")
    print(f"    Distance to Saturate: {satDistance655:.2e} m, Distance to Depletion: {depletionDistance655:.2e} m")

    ## Case Zero Sr 88 enrichment
    goalIsotope = isotopes461[3]
    print(f"\n Step 1: Abundances after First Pass (Targeting {goalIsotope['name']})")
    firstpass_iso461, firstpass461_excited_fraction = calculate_excited_abundances(goalIsotope, isotopes461, linewidth_461, selection_intensity461, Isat_461)
    

    ## Calculate spectra after two passes through 461 pathway
    print("\n--- Case 1: Two Passes through 461nm Ionization Pathway ---")
    goalIsotope = isotopes461[0]  # Target Sr-84 for enrichment

    print(f"\n Step 1: Abundances after First Pass (Targeting {goalIsotope['name']})")
    firstpass_iso461, firstpass461_excited_fraction = calculate_excited_abundances(goalIsotope, isotopes461, linewidth_461, selection_intensity461, Isat_461)
    N_excited_total_step1 = firstpass461_excited_fraction * Nstart
    N_sr84_step1 = N_excited_total_step1 * firstpass_iso461[0]['abundance']
    # print(f"    Number of Excited Sr84 after First Pass: {N_sr84_step1:.2e} atoms, {N_sr84_step1/NstartSr84*100:.2f}% of Original Sr84")
    
    print(f"\n Step 2: Abundances after Second Pass (Targeting {goalIsotope['name']})")
    # The input to this step is the *excited* population from the first pass.
    # The abundances of this population are in `firstpass_iso461`.
    # The total number of atoms in this population is `N_excited_total_step1`.
    secondpass_iso461, secondpass461_excited_fraction = calculate_excited_abundances(goalIsotope, firstpass_iso461, linewidth_461, selection_intensity461, Isat_461)
    N_excited_total_step2 = secondpass461_excited_fraction * N_excited_total_step1
    N_sr84_step2 = N_excited_total_step2 * secondpass_iso461[0]['abundance']
    # print(f"    Number of Excited Sr84 after Second Pass: {N_sr84_step2:.2e} atoms, {N_sr84_step2/NstartSr84*100:.2f}% of Original Sr84")
    

    # Case 2, Deplete Ground State Sr88 using 689nm excitation
    print("\n--- Case 2: Deplete Ground State Sr88 then use 461nm Ionization Pathway ---")
    goalIsotope = isotopes689[3]  # Target Sr-88 for depletion
    print(f"\n Step 1: Abundances after First Pass (Targeting {goalIsotope['name']})")
    firstpass_iso689, firstpass689_unexitedFraction = calculate_unexcited_abundances(goalIsotope, isotopes689, linewidth_689, selection_intensity689, Isat_689)
    NfirstPass689 = firstpass689_unexitedFraction*Nstart*firstpass_iso689[0]['abundance']
    # print(f"    Number of unexcited Sr84 after First Pass: {NfirstPass689:.2e} atoms, {NfirstPass689/NstartSr84*100:.2f}% of Original Sr84")
    isotopes461_depleted = []
    for iso_461, iso_depleted in zip(isotopes461, firstpass_iso689):
        # Sanity check to make sure we're matching isotopes
        if iso_461['name'] != iso_depleted['name']:
            raise ValueError(f"Isotope mismatch: {iso_461['name']} != {iso_depleted['name']}")
        new_iso_data = iso_461.copy() # Start with 461 data (correct shifts)
        new_iso_data['abundance'] = iso_depleted['abundance'] # Overwrite with new abundance
        isotopes461_depleted.append(new_iso_data)

    goalIsotope = isotopes461[0]  # Target Sr-84 for enrichment
    print(f"\n Step 2: Abundances after Second Pass (Targeting {goalIsotope['name']})")
    secondpass_iso461, secondpass461_excited_fraction = calculate_excited_abundances(goalIsotope, isotopes461_depleted, linewidth_461, selection_intensity461, Isat_461)
    N_unexcited_total_step1 = firstpass689_unexitedFraction * Nstart
    N_excited_total_step2 = secondpass461_excited_fraction * N_unexcited_total_step1
    N_sr84_step2 = N_excited_total_step2 * secondpass_iso461[0]['abundance']
    # print(f"    Number of Excited Sr84 after Second Pass: {N_sr84_step2:.2e} atoms, {N_sr84_step2/NstartSr84*100:.2f}% of Original Sr84")


    # Case 3, Deplete Ground State of all non targets isotopes using 689nm excitation
    print("\n--- Case 3: Deplete Ground State of all non targets then use 461nm Ionization Pathway ---")
    goalIsotope = isotopes689[3]  # Target Sr-88 for depletion
    print(f"\n Step 1: Abundances after First Depletion (Targeting {goalIsotope['name']})")
    firstpass_iso689, firstpass689_unexitedFraction = calculate_unexcited_abundances(goalIsotope, isotopes689, linewidth_689, selection_intensity689, Isat_689)
    N_unexcited_total_step1 = firstpass689_unexitedFraction * Nstart
    N_sr84_step1 = N_unexcited_total_step1 * firstpass_iso689[0]['abundance']
    # print(f"    Number of unexcited Sr84 after First Depletion: {N_sr84_step1:.2e} atoms, {N_sr84_step1/NstartSr84*100:.2f}% of Original Sr84")

    goalIsotope = firstpass_iso689[2]  # Target Sr-87 for depletion
    print(f"\n Step 2: Abundances after Second Depletion (Targeting {goalIsotope['name']})")
    secondpass_iso689, secondpass689_unexitedFraction = calculate_unexcited_abundances(goalIsotope, firstpass_iso689, linewidth_689, selection_intensity689, Isat_689)
    N_unexcited_total_step2 = secondpass689_unexitedFraction * N_unexcited_total_step1
    N_sr84_step2 = N_unexcited_total_step2 * secondpass_iso689[0]['abundance']
    # print(f"    Number of unexcited Sr84 after Second Depletion: {N_sr84_step2:.2e} atoms, {N_sr84_step2/NstartSr84*100:.2f}% of Original Sr84")

    goalIsotope = secondpass_iso689[1]  # Target Sr-86 for depletion
    print(f"\n Step 3: Abundances after Third Depletion (Targeting {goalIsotope['name']})")
    thirdpass_iso689, thirdpass689_unexitedFraction = calculate_unexcited_abundances(goalIsotope, secondpass_iso689, linewidth_689, selection_intensity689, Isat_689)
    N_unexcited_total_step3 = thirdpass689_unexitedFraction * N_unexcited_total_step2
    N_sr84_step3 = N_unexcited_total_step3 * thirdpass_iso689[0]['abundance']
    # print(f"    Number of unexcited Sr84 after Third Depletion: {N_sr84_step3:.2e} atoms, {N_sr84_step3/NstartSr84*100:.2f}% of Original Sr84")

    isotopes461_depleted = []
    for iso_461, iso_depleted in zip(isotopes461, thirdpass_iso689):
        # Sanity check to make sure we're matching isotopes
        if iso_461['name'] != iso_depleted['name']:
            raise ValueError(f"Isotope mismatch: {iso_461['name']} != {iso_depleted['name']}")
        new_iso_data = iso_461.copy() # Start with 461 data (correct shifts)
        new_iso_data['abundance'] = iso_depleted['abundance'] # Overwrite with new abundance
        isotopes461_depleted.append(new_iso_data)

    goalIsotope = isotopes461[0]  # Target Sr-84 for enrichment
    print(f"\n Step 4: Abundances after Final 461nm Excitation (Targeting {goalIsotope['name']})")
    finalpass_iso461, finalpass461_excited_fraction = calculate_excited_abundances(goalIsotope, isotopes461_depleted, linewidth_461, selection_intensity461, Isat_461)
    N_excited_total_step4 = finalpass461_excited_fraction * N_unexcited_total_step3
    N_sr84_step4 = N_excited_total_step4 * finalpass_iso461[0]['abundance']
    # print(f"    Number of Excited Sr84 after Final Excitation: {N_sr84_step4:.2e} atoms, {N_sr84_step4/NstartSr84*100:.2f}% of Original Sr84")


    # # Case 4, Deplete Ground State of just largest impurities (86Sr, 86Sr) using 689nm excitation
    # print("\n--- Case 3: Deplete Ground State of all non targets then use 461nm Ionization Pathway ---")
    # goalIsotope = isotopes689[3]  # Target Sr-88 for depletion
    # print(f"\n Step 1: Abundances after First Depletion (Targeting {goalIsotope['name']})")
    # firstpass_iso689, firstpass689_unexitedFraction = calculate_unexcited_abundances(goalIsotope, isotopes689, linewidth_689, selection_intensity689, Isat_689)
    # NfirstPass689 = firstpass689_unexitedFraction*Nstart*firstpass_iso689[0]['abundance']
    # print(f"    Number of unexcited Sr84 after First Depletion: {NfirstPass689:.2e} atoms")
    # goalIsotope = firstpass_iso689[1]  # Target Sr-86 for depletion
    # print(f"\n Step 2: Abundances after Second Depletion (Targeting {goalIsotope['name']})")
    # secondpass_iso689, secondpass689_unexitedFraction = calculate_unexcited_abundances(goalIsotope, firstpass_iso689, linewidth_689, selection_intensity689, Isat_689)
    # NsecondPass689 = secondpass689_unexitedFraction*NfirstPass689*secondpass_iso689[0]['abundance']
    # print(f"    Number of unexcited Sr84 after Second Depletion: {NsecondPass689:.2e} atoms")

    # isotopes461_depleted = []
    # for iso_461, iso_depleted in zip(isotopes461, secondpass_iso689):
    #     # Sanity check to make sure we're matching isotopes
    #     if iso_461['name'] != iso_depleted['name']:
    #         raise ValueError(f"Isotope mismatch: {iso_461['name']} != {iso_depleted['name']}")
    #     new_iso_data = iso_461.copy() # Start with 461 data (correct shifts)
    #     new_iso_data['abundance'] = iso_depleted['abundance'] # Overwrite with new abundance
    #     isotopes461_depleted.append(new_iso_data)

    # goalIsotope = isotopes461[0]  # Target Sr-84 for enrichment
    # print(f"\n Step 4: Abundances after Final 461nm Excitation (Targeting {goalIsotope['name']})")
    # finalpass_iso461, finalpass461_exitedFraction = calculate_excited_abundances(goalIsotope, isotopes461_depleted, linewidth_461, selection_intensity, Isat_461)
    # NfinalPass461 = finalpass461_exitedFraction*NsecondPass689*finalpass_iso461[0]['abundance']
    # print(f"    Number of Excited Sr84 after Final Excitation: {NfinalPass461:.2e} atoms")

if __name__ == "__main__":
    main()