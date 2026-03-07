import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.constants as con
from multiprocessing import Pool

# Follows derivation from "Application of the density matrix method to multiphoton ionization of molecules"
def master_equations(t, y, params):
    """
    Computes the derivatives for the system of ODEs (Eq 4.15 - 4.20).
    
    Args:
        t (float): Time variable (tau).
        y (list): State vector [rho_gg, rho_aa, rho_bb, rho_ag, rho_bg, rho_ba].
        params (dict): Dictionary of physical constants.
    """
    
    # 1. Unpack State Vector
    # y[0..2] are real populations, y[3..5] are complex coherences
    rho_gg = y[0]
    rho_aa = y[1]
    rho_bb = y[2]
    rho_ag = y[3]
    rho_bg = y[4]
    rho_ba = y[5]

    # 2. Unpack Parameters
    rabi_1 = params['rabi_1']
    rabi_2 = params['rabi_2']
    q = params['q']
    gamma_1 = params['gamma_1']
    gamma_2 = params['gamma_2']
    
    # Detunings and Broadening
    del_ag = params['del_ag']
    del_bg = params['del_bg']
    del_ba = params['del_ba']
  

    # 3. Pre-compute Recurring Terms
    # (1 - i/q) and (1 + i/q)
    term_q_minus = (1 - 1j/q)
    term_q_plus  = (1 + 1j/q)

    # 4. Calculate Derivatives (moving LHS terms to RHS)
    
    # Eq 4.15
    d_rho_gg = - 2 * np.imag(rabi_1 * rho_ag) + gamma_1 * rho_aa

    # Eq 4.16
    d_rho_aa = (
         2 * np.imag(rabi_1 * rho_ag)
        - 2 * np.imag(rabi_2 * term_q_minus * rho_ba)
        - gamma_1 * rho_aa
    )

    # Eq 4.17
    d_rho_bb = (
         2 * np.imag(rabi_2 * term_q_plus * rho_ba)
        - gamma_2 * rho_bb
    )

    # Eq 4.18
    # Coeff for rho_ag term
    
    d_rho_ag = (
        -1j*del_ag*rho_ag
        + 1j*rabi_1*(rho_gg - rho_aa)
        + 1j*rabi_2*term_q_minus*rho_bg
    )

    # Eq 4.19
    # Coeff for rho_bg term
    
    d_rho_bg = (
        -1*(1j*del_bg+gamma_2/2)*rho_bg
        +1j*rabi_2*term_q_minus*rho_ag
        -1j*rabi_1*rho_ba
    )

    # Eq 4.20
    # Coeff for rho_ba term

    d_rho_ba = (
        -1*(1j*del_ba+gamma_2/2)*rho_ba
        +1j*rabi_2*term_q_minus*rho_aa
        -1j*rabi_2*term_q_plus*rho_bb
        -1j*rabi_1*rho_bg
    )

    return [d_rho_gg, d_rho_aa, d_rho_bb, d_rho_ag, d_rho_bg, d_rho_ba]

def run_simulation(params, t_span, t_eval):
    """Runs the solver for a single set of parameters."""
    y0 = [1.0, 0.0, 0.0, 0.0+0j, 0.0+0j, 0.0+0j]
    
    sol = solve_ivp(
        fun=master_equations,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='BDF', # Useful for stiff AMO sims where there are large time scale differences
        args=(params,),
        rtol=1e-6,
        atol=1e-9
    )
    return sol

def plot_results(sol):
    t = sol.t
    
    # Extract components
    pop_gg = np.real(sol.y[0])
    pop_aa = np.real(sol.y[1])
    pop_bb = np.real(sol.y[2])
    
    # Magnitudes of coherences
    coh_ag = np.abs(sol.y[3])
    coh_ba = np.abs(sol.y[5])

    # Calculate yield
    yield_t = 1 - pop_gg - pop_aa - pop_bb

    # Create a figure with 3 subplots
    fig, ax1 = plt.figure()
    
    # Plot 1: Populations
    ax1.plot(t, pop_gg, label=r'$\rho_{gg}$')
    ax1.plot(t, pop_aa, label=r'$\rho_{aa}$')
    ax1.plot(t, yield_t, label='Yield (ions)', color='green')
    ax1.set_title("Population Dynamics")
    ax1.set_xlabel(r"Time (s)")
    ax1.set_ylabel("Population")
    ax1.legend(loc='best')
    ax1.grid(True)
    
    plt.tight_layout()
    plt.show()
    
def calculate_stark(i, w):
    # Energy must be divided by hbar to yield angular frequency
    energy_joules = (i * con.e**2) / (2 * con.c * con.epsilon_0 * con.m_e * w**2)
    return energy_joules / con.hbar

def calculate_rabi2(i,w, crossSec, linWidth):
    return 0.5*np.sqrt((crossSec*linWidth*i)/(con.hbar*w))

def calculate_interaction_time(T_celsius=530.0, L_cm=1.5, mass_amu=87.62):
    T_kelvin = T_celsius + 273.15
    L_meters = L_cm / 100.0
    m_kg = mass_amu * con.atomic_mass
    
    v_avg = np.sqrt((8 * con.k * T_kelvin) / (np.pi * m_kg))
    t_int = L_meters / v_avg
    
    return v_avg, t_int

def calculate_ponderomotive_shift(intensity, omega):
    """Calculates the free-electron AC Stark shift (applicable to the continuum/autoionizing state)."""
    energy_joules = (intensity * con.e**2) / (2 * con.c * con.epsilon_0 * con.m_e * omega**2)
    return energy_joules / con.hbar

def calculate_polarizability_shift(intensity, alpha_si):
    """Calculates the AC Stark shift for a bound state given its dynamic polarizability in SI units."""
    energy_joules = - (alpha_si * intensity) / (2 * con.c * con.epsilon_0)
    return energy_joules / con.hbar

# --- CASE 1: Time Dynamics ---
def run_time_dynamics(base_params, w_2, crossPeak, linewidth405):
    print("\n--- Running Case 1: Time Dynamics ---")
    
    t_end = 100e-6
    t_points = 500
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, t_points)

    base_I2 = 5658.84
    i2_values = [1*base_I2, 10*base_I2, 100*base_I2] 
    
    alpha_g_si = -1.0894e-38
    alpha_a_nonres_si = 0.0 

    plt.figure(figsize=(10, 6))

    for val in i2_values:
        current_params = base_params.copy()
        
        # Consistent detuning logic
        shift_g = calculate_polarizability_shift(val, alpha_g_si)
        shift_a = calculate_polarizability_shift(val, alpha_a_nonres_si)
        shift_b = calculate_ponderomotive_shift(val, w_2)

        current_params['del_ag'] = shift_a - shift_g
        current_params['del_ba'] = shift_b - shift_a
        current_params['del_bg'] = current_params['del_ag'] + current_params['del_ba']
        current_params['rabi_2'] = calculate_rabi2(val, w_2, crossPeak, linewidth405)
        
        lbl = f"P_ratio = {val/base_I2:.0f}"
        sol = run_simulation(current_params, t_span, t_eval)

        if sol.success:
            pop_gg = np.real(sol.y[0])
            pop_aa = np.real(sol.y[1])
            pop_bb = np.real(sol.y[2])
            yield_ion = 1.0 - (pop_gg + pop_aa + pop_bb)
            plt.plot(sol.t*1e6, yield_ion, label='Ions, '+lbl)

    plt.title("Case 1: Ion Yield over Time")
    plt.xlabel(r"Time ($\mu$s)")
    plt.ylabel("Ion Population")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Case 2: Power Scaling ---
def run_power_scaling(base_params, w_2, crossPeak, linewidth405, t_int):
    points = 20
    powers = np.logspace(-1, 2, points) 
    yields = []
    bar_length = 20

    t_span = (0, t_int)
    t_eval = [t_int]
    
    beam_radius_m = 0.015 / 2.0
    beam_area_m2 = np.pi * (beam_radius_m)**2
    
    # Pre-calculated 405 nm polarizability for the 5s^2 1S0 ground state
    # From Safronova, M. S., et al. "Blackbody-radiation shift in the Sr optical atomic clock." Physical Review A 87.012509 (2013)
    alpha_g_si = -1.0894e-38
    
    # The non-resonant background polarizability of the 5s5p 1P1 state. 
    # The resonant Fano interaction is handled by the master equations.
    alpha_a_nonres_si = 0.0 

    for i, power in enumerate(powers):
        intensity = power / beam_area_m2
        
        current_params = base_params.copy()
        
        # Ground state and intermediate state detunings driven by bound-state polarizabilities
        shift_g = calculate_polarizability_shift(intensity, alpha_g_si)
        shift_a = calculate_polarizability_shift(intensity, alpha_a_nonres_si)
        current_params['del_ag'] = shift_a - shift_g
        
        # Autoionizing state detuning driven by the free-electron ponderomotive shift
        shift_b = calculate_ponderomotive_shift(intensity, w_2)
        current_params['del_ba'] = shift_b - shift_a
        
        # Total two-photon detuning
        current_params['del_bg'] = current_params['del_ag'] + current_params['del_ba']
        
        current_params['rabi_2'] = calculate_rabi2(intensity, w_2, crossPeak, linewidth405)

        sol = run_simulation(current_params, t_span, t_eval)
        
        if sol.success:
            pop_gg = np.real(sol.y[0, -1])
            pop_aa = np.real(sol.y[1, -1])
            pop_bb = np.real(sol.y[2, -1])
            yield_ion = 1.0 - (pop_gg + pop_aa + pop_bb)
            yields.append(yield_ion)
        else:
            yields.append(0.0)

        progress = (i + 1) / points
        filled = int(bar_length * progress)
        bar = '█' * filled + '-' * (bar_length - filled)
        percent = int(progress * 100)
        
        print(f"\r[{bar}] {percent}% | Step {i + 1}/{points} | P: {power:6.2f} W", end='', flush=True)

    print("\n--- Simulation Complete ---")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(powers, yields, marker='o', linestyle='-', color='blue')
    ax1.set_title("Ionization Yield vs. Autoionization Laser Power")
    ax1.set_xlabel("Laser Power (W)")
    ax1.set_ylabel("Ion Yield")
    ax1.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

def main():
    # Experimental Parameters
    lambda_1 = 460.73330e-9 # From https://physics.nist.gov/PhysRefData/ASD/lines_form.html (Sr-I)
    einstein461 = 2.01e8 # From NIST, [s^-1]
    w_1 = 2*con.pi*con.c/lambda_1
    I1 = 565.884 # 0.1W with 1.5cm spot size
    mu461 = np.sqrt((3*con.pi*con.epsilon_0*con.hbar*einstein461*con.c**3)/(w_1**3))
    rabi461 = (mu461/con.hbar)*np.sqrt((2*I1)/(con.c*con.epsilon_0))
    
    
    lambda_2 = 405.16e-9 # From https://physics.nist.gov/PhysRefData/ASD/lines_form.html (Sr-I)
    linewidth405 = 2*con.pi*45*con.c*100
    w_2 = 2*con.pi*con.c/lambda_2
    I2 = 5658.84 # 1W with 1.5cm diameter spot size
    crossPeak = 5.6e-19 # Peak cross section in m^2
    rabi405 = calculate_rabi2(I2,w_2,crossPeak,linewidth405)

    v_avg, t_int = calculate_interaction_time(T_celsius=530.0, L_cm=1.5)

    base_params = {
        'rabi_1': rabi461,
        'rabi_2': rabi405,
        'q': 6.8,
        'gamma_1': einstein461,
        'gamma_2': linewidth405,
        'del_ag': 0.0, 
        'del_bg': 0.0,
        'del_ba': 0.0,
    }

    # run_time_dynamics(base_params, w_2, crossPeak, linewidth405)
    run_power_scaling(base_params, w_2, crossPeak, linewidth405, t_int)

if __name__ == "__main__":
    main()