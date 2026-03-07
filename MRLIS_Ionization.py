import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.constants as con
from sympy import meijerg, N

# --- Master Equation System ---
def master_equations(t, y, params):
    rho_gg, rho_aa, rho_bb = y[0], y[1], y[2]
    rho_ag, rho_bg, rho_ba = y[3], y[4], y[5]

    rabi_1 = params['rabi_1']
    rabi_2 = params['rabi_2']
    q = params['q']
    gamma_1 = params['gamma_1']
    gamma_2 = params['gamma_2']
    del_ag, del_bg, del_ba = params['del_ag'], params['del_bg'], params['del_ba']

    term_q_minus = (1 - 1j/q)
    term_q_plus  = (1 + 1j/q)

    d_rho_gg = - 2 * np.imag(rabi_1 * rho_ag) + gamma_1 * rho_aa
    d_rho_aa = 2 * np.imag(rabi_1 * rho_ag) - 2 * np.imag(rabi_2 * term_q_minus * rho_ba) - gamma_1 * rho_aa
    d_rho_bb = 2 * np.imag(rabi_2 * term_q_plus * rho_ba) - gamma_2 * rho_bb
    d_rho_ag = -1j*del_ag*rho_ag + 1j*rabi_1*(rho_gg - rho_aa) + 1j*rabi_2*term_q_minus*rho_bg
    d_rho_bg = -(1j*del_bg + gamma_2/2)*rho_bg + 1j*rabi_2*term_q_minus*rho_ag - 1j*rabi_1*rho_ba
    d_rho_ba = -(1j*del_ba + gamma_2/2)*rho_ba + 1j*rabi_2*term_q_minus*rho_aa - 1j*rabi_2*term_q_plus*rho_bb - 1j*rabi_1*rho_bg

    return [d_rho_gg, d_rho_aa, d_rho_bb, d_rho_ag, d_rho_bg, d_rho_ba]

# --- Helper Functions ---
def calculate_rabi2(i, w, crossSec, linWidth):
    return 0.5 * np.sqrt((crossSec * linWidth * i) / (con.hbar * w))

def calculate_polarizability_shift(intensity, alpha_si):
    return -(alpha_si * intensity) / (2 * con.c * con.epsilon_0 * con.hbar)

def calculate_ponderomotive_shift(intensity, omega):
    return (intensity * con.e**2) / (2 * con.c * con.epsilon_0 * con.m_e * omega**2 * con.hbar)

def main():
    # Common Parameters
    lambda_1, lambda_2 = 460.7333e-9, 405.16e-9
    einstein461 = 2.01e8
    crossSi = 5.6e-19 
    d_beam = 0.015
    beam_area = np.pi * (d_beam/2)**2
    T_kelvin = 530 + 273.15
    m_sr = 87.62 * con.atomic_mass
    v_avg = np.sqrt((8 * con.k * T_kelvin) / (np.pi * m_sr))
    t_int = d_beam / v_avg
    
    w_1, w_2 = 2*con.pi*con.c/lambda_1, 2*con.pi*con.c/lambda_2
    gamma_2_auto = 2*con.pi*45*con.c*100 # 45 cm^-1 to rad/s
    
    # Prep Laser Rabi (0.1 W fixed)
    I1 = 0.1 / beam_area
    mu461 = np.sqrt((3*con.pi*con.epsilon_0*con.hbar*einstein461*con.c**3)/(w_1**3))
    rabi461 = (mu461/con.hbar)*np.sqrt((2*I1)/(con.c*con.epsilon_0))

    # Power Array
    powers = np.linspace(0.1, 100, 20)
    yield_me, yield_meijer = [], []

    # Polarizability Constants
    alpha_g_si = -1.0894e-38 

    for p in powers:
        intensity = p / beam_area
        
        # 1. Master Equation Method
        params = {
            'rabi_1': rabi461,
            'rabi_2': calculate_rabi2(intensity, w_2, crossSi, gamma_2_auto),
            'q': 6.8,
            'gamma_1': einstein461,
            'gamma_2': gamma_2_auto,
            'del_ag': calculate_polarizability_shift(intensity, 0) - calculate_polarizability_shift(intensity, alpha_g_si),
            'del_ba': calculate_ponderomotive_shift(intensity, w_2) - calculate_polarizability_shift(intensity, 0),
        }
        params['del_bg'] = params['del_ag'] + params['del_ba']

        sol = solve_ivp(master_equations, (0, t_int), [1, 0, 0, 0j, 0j, 0j], args=(params,), method='BDF')
        yield_me.append(1.0 - np.sum(np.real(sol.y[:3, -1])))

        # 2. Meijer-G Method
        gammaParam = crossSi / (2 * con.hbar * w_2 * d_beam)
        frontTerm = ((gammaParam * p) ** 4) / (16 * np.sqrt(con.pi) * v_avg ** 4)
        tmp = ((gammaParam * p) ** 2) / (4 * v_avg ** 2)
        mG = N(meijerg([[], []], [[-2, -1.5, 0], []], tmp))
        yield_meijer.append(float(1 - frontTerm * mG))

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(powers, yield_me, 'ro-', label='Master Equation (Full AC Stark)')
    plt.plot(powers, yield_meijer, 'k--', label='Meijer-G (Analytical Velocity Avg)')
    plt.xlabel("405nm Laser Power (W)")
    plt.ylabel("Ionization Efficiency")
    plt.title("Comparison of Ionization Models for Strontium")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()