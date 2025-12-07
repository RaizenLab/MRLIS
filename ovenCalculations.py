import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import quad

# General Inputs
kB = 1.38064852e-23  # Boltzmann's constant (m^2 kg s^-2 K^-1)
kBeV = 8.6173324e-5  # Boltzmann's constant (eV/K)
h = 6.62607004e-34   # Planck's Constant (J.s)
me = 9.11e-31        # Electron's mass (kg)
e = 1.6e-19          # Electron charge (C)
epsilon_o = 8.8541878176e-12  # Vacuum permittivity (F/m)
NA = 6.0221e23       # Avogadro's number
A = 6e5              # Richardson's constant (A/m^2 K^2)

# Sr Inputs
DSr = 2 * 0.215e-9   # Strontium atom diameter (m)
Tmelt = 776.9        # Melting temperature (°C)
mSr = 1.4549642e-25  # Sr mass (kg)
M = 87.62            # Atomic weight (g/mol)
Nogrames = 1.4       # Mass of the Strontium rod (g)

# Oven Parameters
Tres = 530           # Reservoir temperature (°C)
Tcap = 630           # Nozzle (Capillary) temperature (°C)
Lcap = 0.02          # Capillary length (m)
dcap = 0.48260e-3    # Diameter of capillary (m)
rcap = dcap / 2      # Capillary radius (m)
beta = (2 * rcap) / Lcap  # Capillary aspect ratio
Ncap = 37            # Number of capillaries

# Vapor Pressure and Mean Free Path
a = 14.232           # Constant for Sr
b = -8572            # Constant for Sr
c = -1.1926          # Constant for Sr

def TOven(temp):
    return temp + 273.15  # Convert °C to Kelvin

def Pvap(temp):
    T = TOven(temp)
    return 10 ** (a + b/T + (c * np.log10(T)))  # Sr vapor pressure (Pa)

def lambda_mf(temp):
    T = TOven(temp)
    return 7.321e-20 * T / (Pvap(temp) * np.pi * (DSr * 100) ** 2)  # Mean free path (cm)

# Plotting Vapor Pressure and Mean Free Path
temps = np.linspace(300, 800, 500)
pvap_vals = [Pvap(t) for t in temps]
lambda_vals = [lambda_mf(t) for t in temps]

fig = plt.figure(figsize=(6.5, 4.875))
gs = GridSpec(1, 2, figure=fig)

# Vapor Pressure Plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogy(temps, pvap_vals, 'b-')
ax1.axvline(Tcap, color='blue', linestyle='--', linewidth=2)
ax1.axhline(Lcap * 100, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('T (°C)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Vapor Pressure (Pa)', fontsize=14, fontweight='bold')
ax1.grid(True)

# Mean Free Path Plot
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(temps, lambda_vals, 'b-')
ax2.axvline(Tcap, color='blue', linestyle='--', linewidth=2)
ax2.axhline(Lcap * 100, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('T (°C)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Mean free path λ (cm)', fontsize=14, fontweight='bold')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Output Flux
def no(temp):
    return Pvap(temp) / (kB * TOven(temp))  # Atomic density in the vapor

def vav(temp):
    return np.sqrt((8 * kB * TOven(temp)) / (mSr * np.pi))  # Average velocity (m/s)

WT = (8 * rcap) / (3 * Lcap)  # Transmission probability
Aorfic = np.pi * (rcap**2)   # Orifice area (m^2)

def Flux1cap(temp):
    return 0.25 * no(temp) * vav(temp) * Aorfic * WT  # Flux output of one capillary

def Fluxtotal(temp):
    return Flux1cap(temp) * Ncap  # Total flux output by all capillaries

# Plotting Atomic Density and Velocity
no_vals = [no(t) for t in temps]
vav_vals = [vav(t) for t in temps]

fig = plt.figure(figsize=(6.5, 4.875))
gs = GridSpec(1, 2, figure=fig)

# Atomic Density Plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogy(temps, no_vals, 'b-')
ax1.set_xlabel('T (°C)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Atomic density', fontsize=14, fontweight='bold')
ax1.grid(True)

# Average Velocity Plot
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(temps, vav_vals, 'b-')
ax2.set_xlabel('T (°C)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Average velocity (m/s)', fontsize=14, fontweight='bold')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Plotting Flux
flux1cap_vals = [Flux1cap(t) for t in temps]
fluxtotal_vals = [Fluxtotal(t) for t in temps]

fig = plt.figure(figsize=(6.5, 4.875))
gs = GridSpec(1, 2, figure=fig)

# Single Capillary Flux
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogy(temps, flux1cap_vals, 'b-')
ax1.set_xlabel('T (°C)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Flux (atoms/sec)', fontsize=14, fontweight='bold')
ax1.grid(True)

# Total Flux
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(temps, fluxtotal_vals, 'b-')
ax2.set_xlabel('T (°C)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Total Flux (atoms/sec)', fontsize=14, fontweight='bold')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Forward Peak Intensity
Ad = 1               # Detector area (cm^2)
ddec = 2             # Distance to detector (cm)

def Ifor(temp):
    return (1 / (4 * np.pi)) * (Ad / ddec ** 2) * no(temp) * vav(temp) * Aorfic

def Nca1(temp):
    return Ifor(temp) / vav(temp)  # Beam density

# Plotting Intensity and Beam Density
ifor_vals = [Ifor(t) for t in temps]
nca1_vals = [Nca1(t) for t in temps]

fig = plt.figure(figsize=(6.5, 4.875))
gs = GridSpec(1, 2, figure=fig)

# Peak Intensity
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogy(temps, ifor_vals, 'b-')
ax1.set_xlabel('T (°C)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Peak Intensity at detector plane\n(atoms/cm² sec)', fontsize=14, fontweight='bold')
ax1.grid(True)

# Beam Density
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(temps, nca1_vals, 'b-')
ax2.set_xlabel('T (°C)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Beam density at the interaction region\n(atoms/cm³)', fontsize=14, fontweight='bold')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Total Working Time
gramSr = NA / M
def Totaltime(temp):
    return (Nogrames * gramSr) / Fluxtotal(temp) / (60 * 60 * 24 * 30)  # Time in months

# Plotting Total Working Time
totaltime_vals = [Totaltime(t) for t in temps]

plt.figure(figsize=(6.5, 4.875))
plt.semilogy(temps, totaltime_vals, 'b-')
plt.xlabel('T (°C)', fontsize=14, fontweight='bold')
plt.ylabel('Total working time (Months)', fontsize=14, fontweight='bold')
plt.grid(True)
plt.show()

# Flux as a Function of Angle
alpha = 0.5 - (1 / (3 * beta ** 2)) * (
    (1 - 2 * beta ** 3 + (2 * beta ** 2 - 1) * np.sqrt(1 + beta ** 2)) /
    (np.sqrt(1 + beta ** 2) - beta ** 2 * np.arcsinh(1 / beta))
)

def q(theta, beta):
    return np.tan(theta) / beta

def R(theta, beta):
    return np.arccos(q(theta,beta))-q(theta,beta)*np.sqrt(1-q(theta,beta)**2)

def j(theta, beta, alpha):
    if np.tan(theta) < beta:
        return alpha * np.cos(theta) + (2 / np.pi) * np.cos(theta) * (
            (1 - alpha) * R(theta, beta) +
            (2 / (3 * q(theta, beta))) * (1 - 2 * alpha) * (1 - (1 - q(theta, beta) ** 2) ** (3/2))
        )
    else:
        return alpha * np.cos(theta) + (4 / (3 * np.pi * q(theta, beta))) * (1 - 2 * alpha) * np.cos(theta)


# Plot Angular Distribution
xx = np.linspace(-1.5, 1.5, 1000)
xx = xx[xx != 0]
j_vals = [j(x, beta, alpha) for x in xx]

plt.figure(figsize=(6.5, 4.875))
plt.plot(xx, j_vals, 'b-')
plt.xlabel('Half angle θ (Radians)', fontsize=14, fontweight='bold')
plt.ylabel('Angular distribution single channel', fontsize=14, fontweight='bold')
plt.grid(True)
plt.show()

# Oven Flux as Function of Angle
temp = Tres
def II(theta):
    return 0.25 * no(temp) * vav(temp) * Aorfic * WT * Ncap * j(theta)

xxx = np.linspace(-1.5, 1.5, 1000)
II_vals = [II(x) for x in xxx]

plt.figure(figsize=(6.5, 4.875))
plt.plot(xxx, II_vals, 'b-')
plt.xlabel('Half angle θ (Radians)', fontsize=14, fontweight='bold')
plt.ylabel('Oven Flux (atoms/s/sr)', fontsize=14, fontweight='bold')
plt.grid(True)
plt.show()

# Final Calculations
theta_max = 0.0195
def integrand(theta):
    return j(theta) * np.sin(theta) * 2 * np.pi

# Integrate over -theta_max to theta_max for the beam flux
FluxToBeam = 0.25 * no(Tres) * vav(Tres) * Aorfic * WT * Ncap * quad(integrand, 0, theta_max, limit=1000, epsabs=1e-8, epsrel=1e-8)[0]
TotalFlux = 0.25 * no(Tres) * vav(Tres) * Aorfic * WT * Ncap
PercentPass = (FluxToBeam / TotalFlux) * 100
TmassToBeam = (FluxToBeam * mSr * 1000) * 86400  # Mass per day (g)
delta_nu = (vav(Tres) / 3e8) * (3e8 / 461e-9) * theta_max * 1e-6  # MHz
Lifetime = (Nogrames * gramSr) / TotalFlux * (1 / 86400)  # Days

print(f"Flux to Beam: {FluxToBeam:.2e} atoms/s")
print(f"Total Flux: {TotalFlux:.2e} atoms/s")
print(f"Percent Pass: {PercentPass:.2f}%")
print(f"Mass to Beam: {TmassToBeam:.2e} g/day")
print(f"Doppler Broadening: {delta_nu:.4f} MHz")
print(f"Lifetime: {Lifetime:.2f} days")