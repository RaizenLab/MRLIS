import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.constants as con

# --- 1. PHYSICS EQUATIONS (CORRECTED) ---
def master_equations_real(t, y, params):
    # 1. Map flattened Real array back to Physical Variables
    rho_gg, rho_aa, rho_bb = y[0], y[1], y[2]
    rho_ag = y[3] + 1j*y[4]
    rho_bg = y[5] + 1j*y[6]
    rho_ba = y[7] + 1j*y[8]
    
    # 2. Unpack Parameters
    rabi_1, rabi_2 = params['rabi_1'], params['rabi_2']
    q = params['q']
    # gamma_1 = spontaneous decay a->g
    # gamma_2 = autoionization rate of b
    # gamma_direct = direct ionization rate from a (background)
    gamma_1, gamma_2 = params['gamma_1'], params['gamma_2']
    gamma_direct = params.get('gamma_direct', 0.0) # NEW TERM
    
    del_ag, del_bg, del_ba = params['del_ag'], params['del_bg'], params['del_ba']

    # The coupling terms for Fano interference
    # (1 - i/q) handles the interference between discrete and continuum paths
    term_q_minus = (1 - 1j/q)
    term_q_plus  = (1 + 1j/q)

    # 3. Calculate Derivatives
    
    # Ground State: Only coupled to a via Rabi 1
    d_rho_gg = - 2 * np.imag(rabi_1 * rho_ag) + gamma_1 * rho_aa

    # Intermediate State |a>:
    # - Gains from g (Rabi 1)
    # - Lost to b (Rabi 2 via continuum, hence term_q)
    # - Decays to g (gamma_1)
    # - NEW: Direct Ionization to continuum (gamma_direct)
    d_rho_aa = (2 * np.imag(rabi_1 * rho_ag)
              - 2 * np.imag(rabi_2 * term_q_minus * rho_ba)
              - gamma_1 * rho_aa
              - gamma_direct * rho_aa) # <--- CRITICAL FIX 

    # Autoionizing State |b>:
    # - Gains from a (Rabi 2)
    # - Decays into continuum (gamma_2)
    d_rho_bb = (2 * np.imag(rabi_2 * term_q_plus * rho_ba)
              - gamma_2 * rho_bb)

    # Coherences (unchanged structure, but physically linked now via gamma_direct)
    d_rho_ag_c = (-1j*del_ag*rho_ag
                + 1j*rabi_1*(rho_gg - rho_aa)
                + 1j*rabi_2*term_q_minus*rho_bg
                - 0.5 * (gamma_1 + gamma_direct) * rho_ag) # Dephasing from decay/ionization

    d_rho_bg_c = (-1*(1j*del_bg + gamma_2/2)*rho_bg
                + 1j*rabi_2*term_q_minus*rho_ag
                - 1j*rabi_1*rho_ba)

    d_rho_ba_c = (-1*(1j*del_ba + (gamma_2 + gamma_1 + gamma_direct)/2)*rho_ba
                + 1j*rabi_2*term_q_minus*rho_aa
                - 1j*rabi_2*term_q_plus*rho_bb
                - 1j*rabi_1*rho_bg)
    
    # We no longer strictly need a "rho_ion" variable because we calculate
    # yield as (1 - sum_populations), which captures both direct and indirect ions.
    d_rho_ion = gamma_2 * rho_bb + gamma_direct * rho_aa

    return [
        np.real(d_rho_gg),    # 0
        np.real(d_rho_aa),    # 1
        np.real(d_rho_bb),    # 2
        np.real(d_rho_ag_c), np.imag(d_rho_ag_c), # 3, 4
        np.real(d_rho_bg_c), np.imag(d_rho_bg_c), # 5, 6
        np.real(d_rho_ba_c), np.imag(d_rho_ba_c), # 7, 8
        np.real(d_rho_ion)    # 9 (Optional tracking)
    ]

# --- 2. UPDATED DEMO: FANO SHAPES ---
def demo_fano_shapes_corrected(base_params, linewidth405):
    print("\n--- Running Demo 1: Fano Interference (Physically Consistent) ---")
    
    # q values to scan
    q_values = [1.0, 2.5, 10.0]
    
    # Scan window for the coupling laser (Scanning over the autoionizing resonance)
    scan_width = 8 * linewidth405 
    detuning_points = np.linspace(-scan_width, scan_width, 200)
    
    scale_factor = 1e12 
    t_scaled_end = 5.0 # 5ps pulse
    
    plt.figure(figsize=(10, 6))

    for q_val in q_values:
        print(f"  Simulating q = {q_val}")
        yields = []
        
        # --- PHYSICAL CONSISTENCY ENFORCEMENT ---
        # The Fano parameter q is roughly proportional to the ratio of 
        # (Discrete Transition) / (Continuum Transition).
        # Therefore, the background rate (gamma_direct) is related to the peak rate (gamma_2)
        # by approximately: gamma_direct ~= gamma_2 / q^2
        
        demo_params = base_params.copy()
        demo_params['q'] = q_val
        demo_params['gamma_direct'] = demo_params['gamma_2'] / (q_val**2) # Scale background
        
        # Moderate power to avoid power broadening washing out the Fano shape
        demo_params['rabi_2'] = 0.5 * linewidth405 
        
        for det in detuning_points:
            local = demo_params.copy()
            
            # Scan Strategy:
            # Laser 1 (Probe) is fixed ON RESONANCE with intermediate state |a>
            local['del_ag'] = 0.0 
            
            # Laser 2 (Coupling) scans across the autoionizing state |b>
            local['del_ba'] = det
            
            # Two-photon detuning tracks the scan
            local['del_bg'] = local['del_ag'] + local['del_ba']
            
            # Rescale
            for k in ['rabi_1', 'rabi_2', 'gamma_1', 'gamma_2', 'gamma_direct', 'del_ag', 'del_bg', 'del_ba']:
                local[k] /= scale_factor
                
            y0 = [1.0] + [0.0]*9
            sol = solve_ivp(
                master_equations_real, (0, t_scaled_end), 
                y0, method='Radau', args=(local,), rtol=1e-4
            )
            
            # --- CORRECT YIELD CALCULATION ---
            # Yield = 1 - (Population remaining in discrete states)
            # This counts ions from BOTH rho_bb (indirect) and rho_aa (direct)
            final_pops = sol.y[:, -1]
            rho_gg, rho_aa, rho_bb = final_pops[0], final_pops[1], final_pops[2]
            total_yield = 1.0 - (rho_gg + rho_aa + rho_bb)
            yields.append(total_yield)
            
        # Normalize and plot
        y_plot = np.array(yields)
        # We normalize by the max of the q=1 case usually, or self-normalize
        y_plot /= np.max(y_plot) 
        
        plt.plot(detuning_points/(2*np.pi*1e12), y_plot, linewidth=2, label=f'q = {q_val}')

    plt.axvline(0, color='k', linestyle='--', alpha=0.3, label="Resonance")
    plt.title("Fano Interference Profiles (Corrected Physics)")
    plt.xlabel(rf"Coupling Laser Detuning (THz)")
    plt.ylabel("Normalized Ion Yield")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.show()

# --- 3. HELPER & MAIN ---
def calculate_rabi2(i,w, crossSec, linWidth):
    return 0.5*np.sqrt((crossSec*linWidth*i)/(con.hbar*w))

def main():
    # Constants
    lambda_1 = 460.73330e-9 
    einstein461 = 2.01e8 
    w_1 = 2*con.pi*con.c/lambda_1
    I1 = 565.884 
    mu461 = np.sqrt((3*con.pi*con.epsilon_0*con.hbar*einstein461*con.c**3)/(w_1**3))
    rabi461 = (mu461/con.hbar)*np.sqrt((2*I1)/(con.c*con.epsilon_0))
    
    lambda_2 = 405.16e-9 
    linewidth405 = 2*con.pi*45*con.c*100 
    w_2 = 2*con.pi*con.c/lambda_2
    I2 = 5658.84 
    crossPeak = 5.6e-19 
    rabi405 = calculate_rabi2(I2, w_2, crossPeak, linewidth405)
    
    base_params = {
        'rabi_1': rabi461,
        'rabi_2': rabi405,
        'q': 5.0,        
        'gamma_1': einstein461,
        'gamma_2': linewidth405, # Autoionization width
        'gamma_direct': 0.0,     # Will be set in demos
        'del_ag': 0.0,
        'del_bg': 0.0,
        'del_ba': 0.0,
    }

    demo_fano_shapes_corrected(base_params, linewidth405)

if __name__ == "__main__":
    main()