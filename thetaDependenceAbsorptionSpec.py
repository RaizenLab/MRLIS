import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as con

# --- Physics Constants for Strontium ---
WAVELENGTH_SR88 = 460.73330e-9
FREQ_0 = con.c / WAVELENGTH_SR88
GAMMA = 2.01e8 
LINEWIDTH = GAMMA / (2 * np.pi) 
MASS_SR = 87.62 * con.atomic_mass 

# --- Experimental Parameters ---
BEAM_DIVERGENCE = 0.02609  # Omega (radians)
TEMP_OVEN = 460 + 273.15   # Kelvin

def get_most_probable_velocity(T, m):
    """Returns vp = sqrt(2kT/m)."""
    return np.sqrt((2 * con.k * T) / m)

def calculate_signal_for_theta(theta_val):
    """
    Calculates the Total Dispersive Signal for a given theta.
    Returns: (freq_axis_GHz, total_signal)
    """
    # 1. Setup Frequency Grid
    freq_axis = np.linspace(FREQ_0 - 0.5e9, FREQ_0 + 0.5e9, 2000)
    
    # 2. Setup Integration Grids (Vectorization)
    vp = get_most_probable_velocity(TEMP_OVEN, MASS_SR)
    v_axis = np.linspace(0.1, 4.0 * vp, 100)
    dv = v_axis[1] - v_axis[0]
    
    alpha_axis = np.linspace(-BEAM_DIVERGENCE/2, BEAM_DIVERGENCE/2, 40)
    dalpha = alpha_axis[1] - alpha_axis[0]

    # 3. Pre-calculate Distribution Weights
    f_v = (2 * v_axis**3 / vp**4) * np.exp(-v_axis**2 / vp**2)
    weight_alpha = (1.0 / BEAM_DIVERGENCE) * dalpha

    # 4. Prepare Broadcast Arrays
    F = freq_axis[:, None, None]
    V = v_axis[None, :, None]
    A = alpha_axis[None, None, :]
    
    # Doppler shifts using the specific THETA input
    doppler_term_plus = FREQ_0 * (V / con.c) * np.sin(theta_val - A)
    doppler_term_minus = FREQ_0 * (V / con.c) * np.sin(-theta_val - A)

    # 5. Isotope Data
    isotopes = [
        {"shift": -270.8e6, "abundance": 0.0056, "name": "Sr-84"},
        {"shift": -124.8e6, "abundance": 0.0986, "name": "Sr-86"},
        {"shift": -51.9e6,  "abundance": 0.023,  "name": "Sr-87, F'=11/2"},
        {"shift": -68.9e6,  "abundance": 0.023,  "name": "Sr-87, F'=9/2"},
        {"shift": -9.7e6,   "abundance": 0.023,  "name": "Sr-87, F'=7/2"},
        {"shift": 0.0,      "abundance": 0.8258, "name": "Sr-88"},
    ]

    total_signal = np.zeros_like(freq_axis)
    half_gamma_sq = (LINEWIDTH / 2.0)**2
    
    # 6. Loop Isotopes and Integrate
    for iso in isotopes:
        w0_iso = FREQ_0 + iso['shift']
        abundance = iso['abundance']
        
        # Lorentzian G(w)
        denom_plus = half_gamma_sq + (F - w0_iso + doppler_term_plus)**2
        denom_minus = half_gamma_sq + (F - w0_iso + doppler_term_minus)**2
        
        G_plus = 1.0 / denom_plus
        G_minus = 1.0 / denom_minus
        
        # Integrate
        integrand_alpha = np.sum((G_plus - G_minus) * weight_alpha, axis=2)
        signal_iso = np.sum(integrand_alpha * f_v[None, :] * dv, axis=1)
        
        total_signal += signal_iso * abundance

    # Return normalized frequency axis (GHz) and signal
    return (freq_axis - FREQ_0)/1e9, total_signal

def main_plot_comparison():
    # Angles to compare (in radians)
    thetas_to_test = [0.01, 0.02, 0.03, 0.035, 0.04]
    
    # Colors for different angles
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    plt.figure(figsize=(12, 7))
    
    # --- Loop over angles and plot ---
    for i, theta in enumerate(thetas_to_test):
        freq_ghz, signal = calculate_signal_for_theta(theta)
        
        label_text = f"Theta = {theta*1000:.1f} mrad"
        plt.plot(freq_ghz, signal, color=colors[i], linewidth=2, label=label_text)

    # --- Add Isotope Markers (Only once) ---
    # We redefine the isotope list just to get the shifts for markers
    isotopes = [
        {"shift": -270.8e6, "name": "Sr-84"},
        {"shift": -124.8e6, "name": "Sr-86"},
        {"shift": -68.9e6,  "name": "Sr-87 F'9/2"},
        {"shift": -51.9e6,  "name": "Sr-87 F'11/2"},
        {"shift": -9.7e6,   "name": "Sr-87 F'7/2"},
        {"shift": 0.0,      "name": "Sr-88"},
    ]
    
    # Get y-limits to place markers nicely
    ymin, ymax = plt.ylim()
    
    for i, iso in enumerate(isotopes):
        shift_ghz = iso['shift'] / 1e9
        plt.axvline(x=shift_ghz, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
        # Place text at the top of the line
        plt.text(shift_ghz, ymax*(-0.15-0.15*i), f"{iso['name']}", rotation=0, fontsize=12, color='black', fontweight='bold', ha='right')

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.xlabel("Detuning from Sr-88 (GHz)", fontsize=12)
    plt.ylabel("Dispersive Signal (arb. units)", fontsize=12)
    plt.title(f"Effect of Laser Angle on Sr Dispersive Signal\n(Atom Beam Divergence = {BEAM_DIVERGENCE*1000:.1f} mrad)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_plot_comparison()