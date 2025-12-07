import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.constants as con

# --- 1. PHYSICS EQUATIONS (Standard) ---
def master_equations_real(t, y, params):
    # 1. Map flattened Real array back to Physical Variables
    rho_gg, rho_aa, rho_bb = y[0], y[1], y[2]
    rho_ag = y[3] + 1j*y[4]
    rho_bg = y[5] + 1j*y[6]
    rho_ba = y[7] + 1j*y[8]
    
    # y[9] is rho_ion (accumulated)

    # 2. Unpack Parameters
    rabi_1, rabi_2 = params['rabi_1'], params['rabi_2']
    q = params['q']
    gamma_1, gamma_2 = params['gamma_1'], params['gamma_2']
    del_ag, del_bg, del_ba = params['del_ag'], params['del_bg'], params['del_ba']

    term_q_minus = (1 - 1j/q)
    term_q_plus  = (1 + 1j/q)

    # 3. Calculate Derivatives
    d_rho_gg = - 2 * np.imag(rabi_1 * rho_ag) + gamma_1 * rho_aa

    d_rho_aa = (2 * np.imag(rabi_1 * rho_ag)
              - 2 * np.imag(rabi_2 * term_q_minus * rho_ba)
              - gamma_1 * rho_aa)

    d_rho_bb = (2 * np.imag(rabi_2 * term_q_plus * rho_ba)
              - gamma_2 * rho_bb)

    d_rho_ag_c = (-1j*del_ag*rho_ag
                + 1j*rabi_1*(rho_gg - rho_aa)
                + 1j*rabi_2*term_q_minus*rho_bg)

    d_rho_bg_c = (-1*(1j*del_bg+gamma_2/2)*rho_bg
                + 1j*rabi_2*term_q_minus*rho_ag
                - 1j*rabi_1*rho_ba)

    d_rho_ba_c = (-1*(1j*del_ba+gamma_2/2)*rho_ba
                + 1j*rabi_2*term_q_minus*rho_aa
                - 1j*rabi_2*term_q_plus*rho_bb
                - 1j*rabi_1*rho_bg)
    
    # Explicit Ion Accumulation
    d_rho_ion = gamma_2 * rho_bb

    # 4. Return Flattened List of 10 derivatives
    return [
        np.real(d_rho_gg),    # 0
        np.real(d_rho_aa),    # 1
        np.real(d_rho_bb),    # 2
        np.real(d_rho_ag_c), np.imag(d_rho_ag_c), # 3, 4
        np.real(d_rho_bg_c), np.imag(d_rho_bg_c), # 5, 6
        np.real(d_rho_ba_c), np.imag(d_rho_ba_c), # 7, 8
        np.real(d_rho_ion)    # 9
    ]

# --- 2. DEMO: FORCED SPLITTING ---
def run_case_2_autler_townes_demo(base_params, linewidth405):
    print("\n--- Running Case 2: Autler-Townes (DEMO MODE) ---")
    print("Overriding parameters to force visible splitting...")
    
    # 1. Setup Parameters (Force high Rabi and High q)
    at_params = base_params.copy()
    
    # FORCE HUGE RABI FREQUENCY (5x Linewidth)
    # This pushes the peaks far apart so they don't merge.
    at_params['rabi_2'] = 5.0 * linewidth405 
    
    # FORCE HIGH Q (Symmetric Fano)
    # This removes the interference that hides one of the peaks.
    at_params['q'] = 100.0 
    
    at_params['del_ba'] = 0.0 
    
    # 2. Scan Range (Wide enough to see both peaks)
    # Scan +/- 1.5 * Rabi
    scan_width = 1.5 * at_params['rabi_2']
    detuning_points = np.linspace(-scan_width, scan_width, 200)
    
    # 3. Time Scaling Factors
    scale_factor = 1e12 
    
    final_yields = []
    
    total_steps = len(detuning_points)
    print(f"  Scanning {total_steps} points...")
    
    for i, det in enumerate(detuning_points):
        if i % 20 == 0:
            print(f"  ... Step {i}/{total_steps}")
            
        local_params = at_params.copy()
        local_params['del_ag'] = det
        local_params['del_bg'] = det + local_params['del_ba']
        
        # Rescaling
        scaled_params = local_params.copy()
        for key in ['rabi_1', 'rabi_2', 'gamma_1', 'gamma_2', 'del_ag', 'del_bg', 'del_ba']:
            scaled_params[key] /= scale_factor
            
        t_scaled_end = 2.0 
        
        y0_real = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        sol = solve_ivp(
            fun=master_equations_real,
            t_span=(0, t_scaled_end),
            y0=y0_real,
            method='Radau', 
            args=(scaled_params,),
            rtol=1e-4, atol=1e-7
        )
        
        total_ions = sol.y[9][-1]
        final_yields.append(total_ions)
        
    print("  ... Done!")

    # 5. Plot Results
    plt.figure(figsize=(10, 6))
    
    x_axis_thz = detuning_points / (2 * np.pi * 1e12)
    y_plot = np.array(final_yields)
    if np.max(y_plot) > 0:
        y_plot = y_plot / np.max(y_plot)
    
    plt.plot(x_axis_thz, y_plot, '-', color='crimson', linewidth=2, label=rf"$\Omega_c$ = 5.0 $\Gamma$ (q=100)")
    
    # Markers
    plt.axvline(0, color='black', linestyle='--', alpha=0.4, label="Resonance")
    
    # Expected Splitting (+/- Rabi/2 in these units)
    # If the user definition of Rabi is 2*V, then peaks are at +/- rabi_2
    expected_pos = at_params['rabi_2'] / (2 * np.pi * 1e12)
    plt.axvline(expected_pos, color='blue', linestyle=':', alpha=0.4, label="Expected Peak")
    plt.axvline(-expected_pos, color='blue', linestyle=':', alpha=0.4)
    
    plt.title("Autler-Townes Splitting (High Power, High q)")
    plt.xlabel(rf"Probe Detuning (THz)")
    plt.ylabel("Normalized Ion Yield")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def demo_fano_shapes_corrected(base_params, linewidth405):
    print("\n--- Running Demo 1: Fano Interference (Scanning Coupling Laser) ---")
    
    # We will simulate the spectrum for different q values
    q_values = [1, 5, 10]
    
    # Scan parameters: Scan the 405nm Coupling Detuning (del_ba)
    scan_width = 10 * linewidth405 
    detuning_points = np.linspace(-scan_width, scan_width, 150)
    
    # Scale factors
    scale_factor = 1e12 
    t_scaled_end = 5 # 2ps integration
    
    plt.figure(figsize=(10, 6))

    for q_val in q_values:
        print(f"  Simulating q = {q_val}")
        yields = []
        
        # Setup Params
        demo_params = base_params.copy()
        demo_params['q'] = q_val                     # <-- q is the varied parameter
        demo_params['rabi_2'] = 1.0 * linewidth405    # Moderate coupling power
        demo_params['del_ag'] = 0.0                   # <-- FIXED Probe ON RESONANCE
        
        # Serial Scan
        for det in detuning_points:
            local = demo_params.copy()
            
            # --- SCAN LOGIC ---
            local['del_ba'] = det                    # <-- SCANNING Coupling Detuning
            local['del_bg'] = local['del_ag'] + det  # Two-photon detuning (del_ag is 0)
            
            # Rescale
            for k in ['rabi_1', 'rabi_2', 'gamma_1', 'gamma_2', 'del_ag', 'del_bg', 'del_ba']:
                local[k] /= scale_factor
                
            sol = solve_ivp(
                master_equations_real, (0, t_scaled_end), 
                [1]+[0]*9, method='Radau', args=(local,), rtol=1e-4
            )
            yields.append(sol.y[9][-1])
            
        # Normalize and plot
        y_plot = np.array(yields)
        y_plot /= np.max(y_plot)
        plt.plot(detuning_points/(2*np.pi*1e12), y_plot, linewidth=2, label=f'q = {q_val}')

    plt.axvline(0, color='k', linestyle='--', alpha=0.3, label="Coupling Resonance")
    

    plt.title("Fano Interference: Scanning Coupling Detuning (405nm)")
    plt.xlabel(rf"Ionization Laser Detuning $\Delta_{{405}}$ (THz)")
    plt.ylabel("Normalized Ionization")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.show()

    return detuning_points, np.array(yields)

# --- DEMO 2: AC STARK SHIFT ---
def demo_ac_stark_shift(base_params, linewidth405):
    print("\n--- Running Demo 2: AC Stark Shift ---")
    
    # We hold the coupling laser (w2) at different DETUNINGS
    # This should shift the probe absorption (w1) peak
    coupling_detunings = [0.1 * linewidth405, 0.5 * linewidth405, 1 * linewidth405]
    
    scan_width = 8 * linewidth405
    detuning_points = np.linspace(-scan_width, scan_width, 150)
    scale_factor = 1e12 
    t_scaled_end = 2.0
    
    plt.figure(figsize=(10, 6))

    for delta_c in coupling_detunings:
        print(f"  Coupling Detuning = {delta_c/1e12:.1f} Trad/s")
        yields = []
        
        demo_params = base_params.copy()
        demo_params['rabi_2'] = 2.0 * linewidth405 # Strong field
        demo_params['del_ba'] = delta_c # <-- OFF RESONANCE COUPLING
        
        for det in detuning_points:
            local = demo_params.copy()
            local['del_ag'] = det
            local['del_bg'] = det + local['del_ba']
            
            for k in ['rabi_1', 'rabi_2', 'gamma_1', 'gamma_2', 'del_ag', 'del_bg', 'del_ba']:
                local[k] /= scale_factor
                
            sol = solve_ivp(
                master_equations_real, (0, t_scaled_end), 
                [1]+[0]*9, method='Radau', args=(local,), rtol=1e-4
            )
            yields.append(sol.y[9][-1])
            
        y_plot = np.array(yields)
        y_plot /= np.max(y_plot)
        lbl = f"$\Delta_{{couple}}$ = 0" if delta_c == 0 else f"$\Delta_{{couple}}$ = {delta_c/(2*np.pi*1e12):.1f} THz"
        plt.plot(detuning_points/(2*np.pi*1e12), y_plot, linewidth=2, label=lbl)

    plt.axvline(0, color='k', linestyle='--', alpha=0.3, label="Bare Resonance")
    plt.title("AC Stark Shift: Displacing Levels with Light")
    plt.xlabel("Probe Detuning (THz)")
    plt.ylabel("Normalized Ionization")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.show()

# --- DEMO 3: POWER BROADENING ---
def demo_power_broadening(base_params, linewidth405):
    print("\n--- Running Demo 3: Power Broadening ---")
    
    # We increase the PROBE intensity (Rabi 1) significantly
    # Note: Rabi 2 (Coupling) is kept weak/moderate to act as a detector
    probe_rabi_factors = [0.1, 0.5, 1] 
    
    # We need to scan the *Probe* detuning
    # But to see broadening, we need to compare widths relative to the natural linewidth
    scan_width = 10 * linewidth405
    detuning_points = np.linspace(-scan_width, scan_width, 150)
    scale_factor = 1e12 
    t_scaled_end = 2.0
    
    plt.figure(figsize=(10, 6))

    for r_factor in probe_rabi_factors:
        print(f"  Probe Power Factor = {r_factor}x")
        yields = []
        
        demo_params = base_params.copy()
        # Vary Probe Strength
        demo_params['rabi_1'] = r_factor * linewidth405 
        # Keep coupling constant
        demo_params['rabi_2'] = 0.5 * linewidth405 
        
        for det in detuning_points:
            local = demo_params.copy()
            local['del_ag'] = det
            local['del_bg'] = det 
            
            for k in ['rabi_1', 'rabi_2', 'gamma_1', 'gamma_2', 'del_ag', 'del_bg', 'del_ba']:
                local[k] /= scale_factor
                
            sol = solve_ivp(
                master_equations_real, (0, t_scaled_end), 
                [1]+[0]*9, method='Radau', args=(local,), rtol=1e-4
            )
            yields.append(sol.y[9][-1])
            
        y_plot = np.array(yields)
        # IMPORTANT: Do not normalize height if you want to see signal growth
        # But for "Broadening", normalizing height helps compare Widths.
        y_plot /= np.max(y_plot) 
        
        plt.plot(detuning_points/(2*np.pi*1e12), y_plot, linewidth=2, label=f'$\Omega_1$ = {r_factor} $\Gamma$')

    plt.title("Power Broadening: Saturation of the Transition")
    plt.xlabel("Detuning (THz)")
    plt.ylabel("Normalized Signal")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.show()

# --- 3. HELPER & MAIN ---
def calculate_stark(i,w):
    return (i*con.e**2)/(2*con.c*con.epsilon_0*con.m_e*w**2)

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
        'q': 6.8,         # This will be overridden in the demo function
        'gamma_1': einstein461,
        'gamma_2': linewidth405,
        'del_ag': 0.0,
        'del_bg': 0.0,
        'del_ba': 0.0,
    }

    # run_case_2_autler_townes_demo(base_params, linewidth405)
    demo_fano_shapes_corrected(base_params, linewidth405)
    demo_ac_stark_shift(base_params, linewidth405)
    demo_power_broadening(base_params, linewidth405)

if __name__ == "__main__":
    main()