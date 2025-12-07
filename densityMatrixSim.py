import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.constants as con
from multiprocessing import Pool

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
    fig, ax1 = plt.figure
    
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
    
def calculate_stark(i,w):
    return (i*con.e**2)/(2*con.c*con.epsilon_0*con.m_e*w**2)

def calculate_rabi2(i,w, crossSec, linWidth):
    return 0.5*np.sqrt((crossSec*linWidth*i)/(con.hbar*w))

# --- CASE 1: Power Scaling ---
def run_case_1_time_dynamics(base_params, w_2, crossPeak, linewidth405):
    print("\n--- Running Case 1: Time Dynamics ---")
    
    # Time settings
    t_end = 100e-6
    t_points = 500
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, t_points)

    # Variations (Ionization Laser Power)
    base_I2 = 5658.84
    i2_values = [1*base_I2, 10*base_I2, 100*base_I2] 

    plt.figure(figsize=(10, 6))

    for val in i2_values:
        # Update Parameters
        current_params = base_params.copy()
        current_params['del_ag'] = calculate_stark(val, w_2)
        current_params['del_ba'] = calculate_stark(val, w_2)
        current_params['rabi_2'] = calculate_rabi2(val, w_2, crossPeak, linewidth405)
        
        lbl = f"P = {val/base_I2:.0f} W"
        print(f"Simulating Time Dynamics: {lbl}")

        sol = run_simulation(current_params, t_span, t_eval)

        if sol.success:
            pop_gg = np.real(sol.y[0])
            pop_aa = np.real(sol.y[1])
            pop_bb = np.real(sol.y[2])
            yield_ion = 1.0 - (pop_gg + pop_aa + pop_bb)

            # Plot Ion Yield only to keep graph clean
            # plt.plot(sol.t*1e6, pop_aa, label=r'$\rho_{aa}$'+lbl, color = 'red')
            plt.plot(sol.t*1e6, yield_ion, label='Ions, '+lbl)
        else:
            print(f"Run failed for {lbl}")

    plt.title("Case 1: Ion Yield over Time")
    plt.xlabel(r"Time ($\mu$s)")
    plt.ylabel("Ion Population")
    plt.legend()
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
    I2 = 5658.84 # 1W with 1.5cm spot size
    crossPeak = 5.6e-19 # Peak cross section in m^2
    rabi405 = calculate_rabi2(I2,w_2,crossPeak,linewidth405)
    
    delStark = calculate_stark(I2,w_2)
    Natom = 1e14
    delT = 340e-6 # Time it takes for an atom to pass through the light interaction field

    # --- 1. CONFIGURATION (Edit Constraints Here) ---
    #  Base dictionary
    base_params = {
        'rabi_1': rabi461,
        'rabi_2': rabi405,
        'q': 6.8,
        'gamma_1': einstein461,
        'gamma_2': linewidth405,
        'del_ag': 0.0,       # Laser 1 on resonance
        'del_bg': 0.0,
        'del_ba': 0.0,
    }

    # run_case_1_time_dynamics(base_params, w_2, crossPeak, linewidth405)

if __name__ == "__main__":
    main()