import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as con

# --- Physics Constants for Strontium ---
WAVELENGTH_SR88 = 460.73330e-9
FREQ_0 = con.c / WAVELENGTH_SR88
GAMMA = 2.01e8 # Einstein A / 2pi (approx linewidth in Hz * 2pi) -> roughly 32 MHz
LINEWIDTH = GAMMA / (2 * np.pi) # Natural linewidth in Hz
MASS_SR = 87.62 * con.atomic_mass # Average mass approx

# --- Experimental Parameters ---
THETA_LASER = 0.0351       # Theta (radians)
BEAM_DIVERGENCE = 0.02609  # Omega (radians)
TEMP_OVEN = 460 + 273.15   # Kelvin

def get_most_probable_velocity(T, m):
    """Returns vp = sqrt(2kT/m)."""
    return np.sqrt((2 * con.k * T) / m)

def calculate_sr_spectrum():
    # 1. Setup Frequency Grid
    # Range: +/- 1.5 GHz around resonance to see all isotopes
    freq_axis = np.linspace(FREQ_0 - 0.5e9, FREQ_0 + 0.5e9, 2000)
    
    # 2. Setup Integration Grids (Vectorization)
    # Velocity Grid: 0 to 4*vp
    vp = get_most_probable_velocity(TEMP_OVEN, MASS_SR)
    v_axis = np.linspace(0.1, 4.0 * vp, 100) # Integration steps for velocity
    dv = v_axis[1] - v_axis[0]
    
    # Alpha Grid: -Omega/2 to +Omega/2
    alpha_axis = np.linspace(-BEAM_DIVERGENCE/2, BEAM_DIVERGENCE/2, 40)
    dalpha = alpha_axis[1] - alpha_axis[0]

    # 3. Pre-calculate Distribution Weights
    # Eq [7]: f(v) = (2v^3 / vp^4) * exp(-v^2/vp^2)
    # normalized such that integral(f(v)dv) = 1
    f_v = (2 * v_axis**3 / vp**4) * np.exp(-v_axis**2 / vp**2)
    
    # Eq [6]: f(alpha) = 1/Omega
    weight_alpha = (1.0 / BEAM_DIVERGENCE) * dalpha

    # 4. Prepare Broadcast Arrays (3D: Freq x Vel x Alpha)
    # shape: (N_freq, N_vel, N_alpha)
    F = freq_axis[:, None, None]
    V = v_axis[None, :, None]
    A = alpha_axis[None, None, :]
    
    # Pre-calculate Doppler shifts (depend on v and alpha, not isotope)
    # Shift = w0 * (v/c) * sin(theta - alpha)
    # Using approx w0 = FREQ_0 for the Doppler term is sufficient (difference is < 1ppm)
    doppler_term_plus = FREQ_0 * (V / con.c) * np.sin(THETA_LASER - A)
    doppler_term_minus = FREQ_0 * (V / con.c) * np.sin(-THETA_LASER - A)

    # 5. Isotope Data
    isotopes = [
        {"shift": -270.8e6, "abundance": 0.0056, "name": "Sr-84"},
        {"shift": -124.8e6, "abundance": 0.0986, "name": "Sr-86"},
        {"shift": -51.9e6,  "abundance": 0.023, "name": "Sr-87, F'=11/2"},
        {"shift": -68.9e6,  "abundance": 0.023, "name": "Sr-87, F'=9/2"},
        {"shift": -9.7e6,  "abundance": 0.023, "name": "Sr-87, F'=7/2"},
        {"shift": 0.0,      "abundance": 0.8258, "name": "Sr-88"},
    ]

    total_signal = np.zeros_like(freq_axis)
    
    plt.figure(figsize=(10, 6))
    
    # 6. Loop Isotopes and Integrate
    half_gamma_sq = (LINEWIDTH / 2.0)**2
    
    for i, iso in enumerate(isotopes):
        w0_iso = FREQ_0 + iso['shift']
        abundance = iso['abundance']
        name = iso['name']
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        # Eq [5]: Lorentzian G(w)
        # Denom = (Gamma/2)^2 + (w - w0 + shift)^2
        # We broadcast F (freq array) against the doppler shifts
        denom_plus = half_gamma_sq + (F - w0_iso + doppler_term_plus)**2
        denom_minus = half_gamma_sq + (F - w0_iso + doppler_term_minus)**2
        
        G_plus = 1.0 / denom_plus
        G_minus = 1.0 / denom_minus
        
        # Eq [4]: Integral F = integral( [G+ - G-] * f(v) * f(alpha) )
        # Sum over alpha first
        integrand_alpha = np.sum((G_plus - G_minus) * weight_alpha, axis=2)
        
        # Sum over velocity
        # integrand_alpha is now (N_freq, N_vel)
        # multiply by f(v) * dv and sum over velocity axis
        signal_iso = np.sum(integrand_alpha * f_v[None, :] * dv, axis=1)
        
        # Add to total
        weighted_signal = signal_iso * abundance
        total_signal += weighted_signal
        
        # Plot individual lines
        label_text = f"{name} ({abundance*100:.2f}%) [{iso['shift']/1e6:.1f} MHz]"
        plt.axvline(x=iso['shift'] / 1e9, color=colors[i], linestyle='--', linewidth=1, label=label_text, alpha=0.8)
        plt.plot((freq_axis - FREQ_0)/1e9, weighted_signal, color=colors[i], linewidth=1.5,
                 alpha=0.6, linestyle='--')

    # 7. Final Plotting
    plt.plot((freq_axis - FREQ_0)/1e9, total_signal, 'k-', linewidth=2, label='Total Signal')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
    
    plt.xlabel("Detuning from Sr-88 (GHz)")
    plt.ylabel("Dispersive Signal (arb. units)")
    # plt.title(f"Sr Beam Dispersive Signal\nTheta={THETA_LASER*1000:.1f} mrad, Div={BEAM_DIVERGENCE*1000:.1f} mrad")
    plt.title(f"Sr Beam Dispersive Signal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    calculate_sr_spectrum()