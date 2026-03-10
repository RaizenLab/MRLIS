import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.constants as con
from joblib import Parallel, delayed
from sympy import meijerg, N

# Follows derivation from "Application of the density matrix method to multiphoton ionization of molecules"
def master_equations(t, y, params):
    """
    Computes the derivatives for the system of ODEs (Eq 4.15 - 4.20).
    
    Args:
        t (float): Time variable (tau).
        y (list): State vector [rho_gg, rho_aa, rho_bb, rho_ag, rho_bg, rho_ba].
        params (dict): Dictionary of physical constants.
    """
    
    # # 1. Unpack State Vector
    # # y[0..2] are real populations, y[3..5] are complex coherences
    # rho_gg = y[0]
    # rho_aa = y[1]
    # rho_bb = y[2]
    # rho_ag = y[3]
    # rho_bg = y[4]
    # rho_ba = y[5]
    rho_gg = y[0]
    rho_aa = y[1]
    rho_bb = y[2]
    
    # Reconstruct complex coherences
    rho_ag = y[3] + 1j * y[4]
    rho_bg = y[5] + 1j * y[6]
    rho_ba = y[7] + 1j * y[8]

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
    
    # --- Eq 4.15 ---
    d_rho_gg = - 2 * np.imag(rabi_1 * rho_ag) + gamma_1 * rho_aa

    # --- Eq 4.16 ---
    d_rho_aa = (
         2 * np.imag(rabi_1 * rho_ag)
        - 2 * np.imag(rabi_2 * term_q_minus * rho_ba)
        - gamma_1 * rho_aa
    )

    # --- Eq 4.17 ---
    d_rho_bb = (
         2 * np.imag(rabi_2 * term_q_plus * rho_ba)
        - gamma_2 * rho_bb
    )

    # --- Eq 4.18 ---
    # Coeff for rho_ag term
    # without spontaneous emission
    # d_rho_ag = (
    #     -1j*del_ag*rho_ag
    #     + 1j*rabi_1*(rho_gg - rho_aa)
    #     + 1j*rabi_2*term_q_minus*rho_bg
    # )

    # with spontaneous emission as per Metcalf Laser Cooling and Trapping
    d_rho_ag = (
        -(1j * del_ag + gamma_1 / 2)*rho_ag
        + 1j*rabi_1*(rho_gg - rho_aa)
        + 1j*rabi_2*term_q_minus*rho_bg
    )

    # --- Eq 4.19 ---
    # Coeff for rho_bg term
    # without spontaneous emission
    # d_rho_bg = (
    #     -1*(1j*del_bg+gamma_2/2)*rho_bg
    #     +1j*rabi_2*term_q_minus*rho_ag
    #     -1j*rabi_1*rho_ba
    # )

    # with spontaneous emission as per Metcalf Laser Cooling and Trapping
    d_rho_bg = (
        -(1j * del_bg + gamma_2 / 2)*rho_bg
        +1j*rabi_2*term_q_minus*rho_ag
        -1j*rabi_1*rho_ba
    )

    # --- Eq 4.20 ---
    # Coeff for rho_ba term
    # without spontaneous emission
    # d_rho_ba = (
    #     -1*(1j*del_ba+gamma_2/2)*rho_ba
    #     +1j*rabi_2*term_q_minus*rho_aa
    #     -1j*rabi_2*term_q_plus*rho_bb
    #     -1j*rabi_1*rho_bg
    # )

    # with spontaneous emission as per Metcalf Laser Cooling and Trapping
    d_rho_ba = (
        -(1j * del_ba + (gamma_1 + gamma_2) / 2) * rho_ba
        + 1j * rabi_2 * term_q_minus * rho_aa
        - 1j * rabi_2 * term_q_plus * rho_bb
        - 1j * rabi_1 * rho_bg
    )

    # return [d_rho_gg, d_rho_aa, d_rho_bb, d_rho_ag, d_rho_bg, d_rho_ba]
    return [
        np.real(d_rho_gg),
        np.real(d_rho_aa),
        np.real(d_rho_bb),
        np.real(d_rho_ag),
        np.imag(d_rho_ag),
        np.real(d_rho_bg),
        np.imag(d_rho_bg),
        np.real(d_rho_ba),
        np.imag(d_rho_ba)
    ]

def run_simulation(params, t_span, t_eval):
    """Runs the solver for a single set of parameters."""
    # y0 = [1.0, 0.0, 0.0, 0.0+0j, 0.0+0j, 0.0+0j]
    y0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # sol = solve_ivp(
    #     fun=master_equations,
    #     t_span=t_span,
    #     y0=y0,
    #     t_eval=t_eval,
    #     method='BDF', # Useful for stiff AMO sims where there are large time scale differences
    #     args=(params,),
    #     rtol=1e-6,
    #     atol=1e-9
    # )
    sol = solve_ivp(
        fun=master_equations,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='Radau',
        args=(params,),
        rtol=1e-6,
        atol=1e-9,
        first_step=1e-12
    )
    return sol

def run_analytical_scaling(config, powers, mass_amu=87.62):
    """
    Calculates the analytical ionization yield using the Meijer G-function
    for a thermal atomic beam traversing a Gaussian laser profile.
    """
    crossSi = config['crossPeak']
    w_2 = config['w_2']
    
    # Reconstruct beam diameter from the stored area
    beam_radius_m = np.sqrt(config['beam_area_m2'] / np.pi)
    d_beam = 2.0 * beam_radius_m
    
    # Retrieve thermal velocity
    v_avg, _ = calculate_interaction_time(T_celsius=530.0, L_cm=1.5, mass_amu=mass_amu)
    
    gammaParam = crossSi / (2.0 * con.hbar * w_2 * d_beam)
    
    yield_meijer = []
    for p in powers:
        frontTerm = ((gammaParam * p) ** 4) / (16.0 * np.sqrt(con.pi) * v_avg ** 4)
        tmp = ((gammaParam * p) ** 2) / (4.0 * v_avg ** 2)
        
        # Evaluate Meijer G-function
        mG = float(N(meijerg([[], []], [[-2, -1.5, 0], []], tmp)))
        yield_meijer.append(1.0 - frontTerm * mG)
        
    return np.array(yield_meijer)

# Pretty sure this is depricated but keeping it for now
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

def simulate_single_power(power, config, detuningSelect):
    intensity = power / config['beam_area_m2']
    current_params = config['base_params'].copy()
    
    shift_g = calculate_polarizability_shift(intensity, config['alpha_g_si'])
    shift_a = calculate_polarizability_shift(intensity, config['alpha_a_nonres_si'])
    
    current_params['del_ag'] = detuningSelect + (shift_a - shift_g)
    
    shift_b = calculate_ponderomotive_shift(intensity, config['w_2'])
    current_params['del_ba'] = shift_b - shift_a
    current_params['del_bg'] = current_params['del_ag'] + current_params['del_ba']
    
    current_params['rabi_2'] = calculate_rabi2(
        intensity, 
        config['w_2'], 
        config['crossPeak'], 
        config['linewidth405']
    )

    sol = run_simulation(current_params, config['t_span'], config['t_eval'])
    
    if sol.success:
        pop_gg = np.real(sol.y[0, -1])
        pop_aa = np.real(sol.y[1, -1])
        pop_bb = np.real(sol.y[2, -1])
        return 1.0 - (pop_gg + pop_aa + pop_bb)
    
    return 0.0

# --- Case 1: Power Scaling of Ionization Efficiency ---
# No isotope shift just a single atom
def run_power_scaling_parallel(config, plotVal, finalEff):
    points = 20
    powers = np.logspace(-1, 2, points) 
    yields = Parallel(n_jobs=-1, verbose=10)(delayed(simulate_single_power)(power,config,0) for power in powers)
    
    if plotVal:
        print("\n--- Simulation Complete ---")
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(powers, yields, marker='o', linestyle='-', color='blue')
        ax1.set_title("Ionization Yield vs. Autoionization Laser Power")
        ax1.set_xlabel("Laser Power (W)")
        ax1.set_ylabel("Ion Yield")
        ax1.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()
    
    if finalEff:
        print(f'Ionization probability of single Sr atom at 100W: {yields[-1]:.6f}')

    return powers, np.array(yields)

def run_single_iso(config, iso):
    isoConfig = config.copy()  
    Natom = 1e14
    intensity1 = isoConfig['P1']/isoConfig['beam_area_m2'] 
    
    w_1Iso = isoConfig['w_1'] + 2 * np.pi * iso["shift"]
    
    # Detuning between the Sr-84 locked laser and the evaluated isotope
    detuningSelect = isoConfig['w_laser'] - w_1Iso
    
    mu461 = np.sqrt((3 * con.pi * con.epsilon_0 * con.hbar * isoConfig['Einstein1'] * con.c**3) / (w_1Iso**3))
    rabi461 = (mu461 / con.hbar) * np.sqrt((2 * intensity1) / (con.c * con.epsilon_0))
    
    iso_params = isoConfig['base_params'].copy()
    iso_params['rabi_1'] = rabi461
    
    config['base_params'] = iso_params

    v_avg, t_int = calculate_interaction_time(T_celsius=530.0, L_cm=1.5, mass_amu=iso["mass"])


    yield_ion = simulate_single_power(isoConfig['P2'], config, detuningSelect)
    return yield_ion


def main():
    # Experimental Parameters
    # --- Beam Shape ---
    beam_radius_m = 0.015 / 2.0
    beam_area_m2 = np.pi * (beam_radius_m)**2 # 1.5cm diameter spot size

    # --- Laser 1 ---
    lambda_1 = 460.73330e-9 # From https://physics.nist.gov/PhysRefData/ASD/lines_form.html (Sr-I)
    einstein461 = 2.01e8 # From NIST, [s^-1]
    w_1 = 2*con.pi*con.c/lambda_1
    P1 = 0.1 # 0.1 W
    I1 = P1/beam_area_m2 # 0.1W with 1.5cm spot size
    mu461 = np.sqrt((3*con.pi*con.epsilon_0*con.hbar*einstein461*con.c**3)/(w_1**3))
    rabi461 = (mu461/con.hbar)*np.sqrt((2*I1)/(con.c*con.epsilon_0))
    
    # --- Laser 2 ---
    lambda_2 = 405.16e-9 # From https://physics.nist.gov/PhysRefData/ASD/lines_form.html (Sr-I)
    linewidth405 = 2*con.pi*45*con.c*100
    w_2 = 2*con.pi*con.c/lambda_2
    P2 = 1 # 1W
    I2 = P2/beam_area_m2 # 1W with 
    crossPeak = 5.6e-19 # Peak cross section in m^2
    rabi405 = calculate_rabi2(I2,w_2,crossPeak,linewidth405)

    # --- Isotope Data ---
    isotopes = [
        {"mass": 83.913419, "shift": -270.8e6, "abundance": 0.0056, "name": "Sr-84"},
        {"mass": 85.90926073, "shift": -124.8e6, "abundance": 0.0986, "name": "Sr-86"},
        {"mass": 86.90887750, "shift": -68.9e6, "abundance": 0.0700, "name": "Sr-87"},
        {"mass": 87.90561226, "shift": 0.0, "abundance": 0.8258, "name": "Sr-88"},
    ]
    shift_sr84 = next(iso["shift"] for iso in isotopes if iso["name"] == "Sr-84") # Used in Selectivity Calc, where Sr84 is the target
    w_laser = w_1 + 2 * np.pi * shift_sr84

    # --- Atom Polarizability ---
    # Pre-calculated 405 nm polarizability for the 5s^2 1S0 ground state
    # From Safronova, M. S., et al. "Blackbody-radiation shift in the Sr optical atomic clock." Physical Review A 87.012509 (2013)
    alpha_g_si = -1.0894e-38
    
    # The non-resonant background polarizability of the 5s5p 1P1 state. 
    # The resonant Fano interaction is handled by the master equations.
    alpha_a_nonres_si = 0.0 

    v_avg, t_int = calculate_interaction_time(T_celsius=530.0, L_cm=1.5, mass_amu=87.62)

    t_span = (0, t_int)
    t_eval = [t_int]

    # --- Parameters for OBE Simulation ---
    base_params = {
        'rabi_1': rabi461,
        'rabi_2': rabi405,
        'q': 6.8,
        'gamma_1': einstein461/(2*np.pi),
        'gamma_2': linewidth405,
        'del_ag': 0.0, 
        'del_bg': 0.0,
        'del_ba': 0.0,
    }

    # --- Experiment Config ---
    config = {
        'beam_area_m2': beam_area_m2,
        'base_params': base_params,
        'alpha_g_si': alpha_g_si,
        'alpha_a_nonres_si': alpha_a_nonres_si,
        'w_laser':w_laser,
        'w_1': w_1,
        'w_2': w_2,
        'P1': P1,
        'P2': P2,
        'Einstein1':einstein461,
        'linewidth405': linewidth405,
        'crossPeak': crossPeak,
        't_span': t_span,
        't_eval': t_eval
        }
    
    # run_time_dynamics(base_params, w_2, crossPeak, linewidth405)
    # run_power_scaling(base_params, w_2, crossPeak, linewidth405, t_int)
    # run_power_scaling_parallel(config, False, True)

    # --- Compare with Analytical Result from Rochester Scientific ---
    powers, yield_me = run_power_scaling_parallel(config, False, False)
    yield_meijer = run_analytical_scaling(config, powers)

    plt.figure(figsize=(8, 5))
    plt.plot(powers, yield_me, 'ro-', label='Master Equation (Full AC Stark)')
    plt.plot(powers, yield_meijer, 'k--', label='Meijer-G (Analytical Velocity Avg)')
    plt.xlabel("405nm Laser Power (W)")
    plt.ylabel("Ionization Efficiency")
    plt.title("Comparison of Ionization Models for Strontium")
    plt.legend()
    plt.grid(True)
    plt.show()
    isoConfig = config.copy()
    isoConfig['P2'] = 100 # 100W testing for selectivity

    # --- Unit Comparison with standard power scaling ---
    # iso84 = next(iso for iso in isotopes if iso["name"] == "Sr-84")
    # yield84 = run_single_iso(isoConfig,iso84)
    # print(f'Ionization Probability of Sr84 when targeted: {yield84:.6f}')
    # iso88 = next(iso for iso in isotopes if iso["name"] == "Sr-88")
    # yield88 = run_single_iso(isoConfig,iso88)
    # print(f'Ionization Probability of Sr88 when targeted: {yield88:.6f}')

if __name__ == "__main__":
    main()