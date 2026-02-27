import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as con

# # --- Physics Constants for Strontium 461nm---
# WAVELENGTH_SR88 = 460.73330e-9
# FREQ_0 = con.c / WAVELENGTH_SR88
# GAMMA = 2.01e8
# LINEWIDTH = GAMMA / (2 * np.pi)
# MASS_SR = 87.62 * con.atomic_mass

# --- Physics Constants for Strontium+ 421nm---
WAVELENGTH_SR88 = 421.5524e-9
FREQ_0 = con.c / WAVELENGTH_SR88
GAMMA = 1.279e8
LINEWIDTH = GAMMA / (2 * np.pi)
MASS_SR = 87.62 * con.atomic_mass

# --- Experimental Parameters ---
THETA_LASER = 0.0351
BEAM_DIVERGENCE = 0.02609
TEMP_OVEN = 460 + 273.15

def get_most_probable_velocity(T, m):
    return np.sqrt((2 * con.k * T) / m)

def calculate_absorption_spectrum(use_effusive=False, v_mean=55.6, v_std=26.5):
    freq_axis = np.linspace(FREQ_0 - 1.5e9, FREQ_0 + 1.5e9, 4000)
    half_gamma_sq = (LINEWIDTH / 2.0)**2
    
    if use_effusive:
        vp = get_most_probable_velocity(TEMP_OVEN, MASS_SR)
        v_axis = np.linspace(0.1, 4.0 * vp, 200)
        dv = v_axis[1] - v_axis[0]
        f_v = (2 * v_axis**3 / vp**4) * np.exp(-v_axis**2 / vp**2)
        f_v /= np.sum(f_v * dv)
        
        alpha_axis = np.linspace(-BEAM_DIVERGENCE/2, BEAM_DIVERGENCE/2, 40)
        dalpha = alpha_axis[1] - alpha_axis[0]
        weight_alpha = (1.0 / BEAM_DIVERGENCE) * dalpha
        
        F_3d = freq_axis[:, None, None]
        V_3d = v_axis[None, :, None]
        A_3d = alpha_axis[None, None, :]
        
        doppler_term = FREQ_0 * (V_3d / con.c) * np.sin(THETA_LASER - A_3d)
    else:
        v_min = v_mean - 4.0 * v_std
        v_max = v_mean + 4.0 * v_std
        v_axis = np.linspace(v_min, v_max, 400)
        dv = v_axis[1] - v_axis[0]
        
        f_v = (1.0 / (v_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((v_axis - v_mean) / v_std)**2)
        f_v /= np.sum(f_v * dv)
        
        F_2d = freq_axis[:, None]
        V_2d = v_axis[None, :]
        
        doppler_term = FREQ_0 * (V_2d / con.c)

    # # 4. Isotope Data 461nm Line Natural Abundance
    # isotopes = [
    #     {"shift": -270.8e6, "abundance": 0.0056, "name": "Sr-84"},
    #     {"shift": -124.8e6, "abundance": 0.0986, "name": "Sr-86"},
    #     {"shift": -51.9e6,  "abundance": 0.023,  "name": "Sr-87, F'=11/2"},
    #     {"shift": -68.9e6,  "abundance": 0.023,  "name": "Sr-87, F'=9/2"},
    #     {"shift": -9.7e6,   "abundance": 0.023,  "name": "Sr-87, F'=7/2"},
    #     {"shift": 0.0,      "abundance": 0.8258, "name": "Sr-88"},
    # ]
    # 4. Isotope Data 421nm Line Single Pass Enriched Abundance
    isotopes = [
        {"shift": -378e6, "abundance": 0.3862, "name": "Sr-84"},
        {"shift": -170.8e6, "abundance": 0.1594, "name": "Sr-86"},
        {"shift": -60e6,  "abundance": 0.0598,  "name": "Sr-87"},
        {"shift": 0.0,      "abundance": 0.3946, "name": "Sr-88"},
    ]

    total_signal = np.zeros_like(freq_axis)
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    for i, iso in enumerate(isotopes):
        w0_iso = FREQ_0 + iso['shift']
        abundance = iso['abundance']
        name = iso['name']
        
        if use_effusive:
            denom = half_gamma_sq + (F_3d - w0_iso + doppler_term)**2
            G = 1.0 / denom
            integrand_alpha = np.sum(G * weight_alpha, axis=2)
            signal_iso = np.sum(integrand_alpha * f_v[None, :] * dv, axis=1)
        else:
            denom = half_gamma_sq + (F_2d - w0_iso + doppler_term)**2
            G = 1.0 / denom
            signal_iso = np.sum(G * f_v[None, :] * dv, axis=1)
            
        weighted_signal = signal_iso * abundance
        total_signal += weighted_signal
        
        label_text = f"{name} ({abundance*100:.2f}%) [{iso['shift']/1e6:.1f} MHz]"
        plt.axvline(x=iso['shift'] / 1e9, color=colors[i], linestyle='--', linewidth=1, label=label_text, alpha=0.8)
        plt.plot((freq_axis - FREQ_0)/1e9, weighted_signal, color=colors[i], linewidth=1.5, alpha=0.6, linestyle='-')

    plt.plot((freq_axis - FREQ_0)/1e9, total_signal, 'k-', linewidth=2, label='Total Absorption')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
    
    plt.xlabel("Detuning from Sr-88 (GHz)")
    plt.ylabel("Absorption Signal (arb. units)")
    
    if use_effusive:
        title_str = "Sr Beam Absorption Profile (Effusive Model)"
    else:
        title_str = f"Sr Beam Absorption Profile (Transverse Gaussian: \u03bc={v_mean} m/s, \u03c3={v_std} m/s)"
    
    plt.title(title_str)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # calculate_absorption_spectrum(use_effusive=True)
    # Hand Tuned results
    # calculate_absorption_spectrum(use_effusive=False, v_mean=55.6, v_std=26.5)
    # PSO Optimized Results
    calculate_absorption_spectrum(use_effusive=False, v_mean=-16.9, v_std=9.9)