import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the exponential approach function for heating (approaching T_ss)
def exp_approach(t, T_ss, A, tau):
    return T_ss + A * np.exp(-t / tau)

# Define the exponential decay function for cooling (approaching T_amb)
def exp_decay(t, T_amb, A, tau):
    return T_amb + A * np.exp(-t / tau)

# Function to find segments where power is constant
def find_constant_p_segments(times, powers, temps, min_points=4):
    segments = []
    start = 0
    for i in range(1, len(times)):
        if powers[i] != powers[start]:
            if i - start >= min_points:
                segments.append((start, i - 1))
            start = i
    if len(times) - start >= min_points:
        segments.append((start, len(times) - 1))
    return segments

# Parse heating data
heating_data = """
0 43.29 37.38 29.36 29.64
5 43.29 54.66 29.36 86.25
10 43.29 71.8 29.36 113.8
15 43.29 86.76 29.36 129.7
20 43.29 101.4 29.36 142.8
25 43.29 115 29.36 154.4
30 43.29 128 29.36 165.6
35 43.29 138.2 29.36 175.2
40 43.29 146.8 29.36 183
45 43.29 154.7 29.36 189.4
50 43.29 161.2 29.36 194.7
55 43.29 168.9 29.36 201
60 43.29 174.7 44.7 206.3
65 43.29 180.8 44.7 217
70 43.29 186.8 64.08 224.3
75 43.29 193.6 64.08 234.4
80 43.29 200.4 93.96 242.9
85 43.29 210.5 93.96 258.9
90 139.66 219.8 93.96 270.2
95 139.66 254 93.96 307
100 139.66 285.6 93.96 344.3
105 139.66 313.9 93.96 358.7
110 139.66 339.1 93.96 381.9
115 139.66 358.4 93.96 403
120 139.66 372.8 93.96 416.2
125 139.66 385.2 93.96 427
130 139.66 395.6 93.96 435.4
135 139.66 404 93.96 442
140 139.66 410.6 93.96 447.1
145 139.66 416.3 93.96 451.6
150 172.3167 421.4 93.96 455.5
151 172.3167 424.3 93.96 457.3
152 172.3167 426.6 93.96 459
153 172.3167 428.8 93.96 460.4
154 172.3167 431 93.96 462
155 139.9398 433.2 93.96 463.5
160 139.9398 436 93.96 465.7
"""
lines = heating_data.strip().split('\n')
heating_times = []
Pcen_h = []
Tcen_h = []
Pnoz_h = []
Tnoz_h = []
for line in lines:
    parts = line.split()
    heating_times.append(float(parts[0]))
    Pcen_h.append(float(parts[1]))
    Tcen_h.append(float(parts[2]))
    Pnoz_h.append(float(parts[3]))
    Tnoz_h.append(float(parts[4]))

heating_times = np.array(heating_times)
Pcen_h = np.array(Pcen_h)
Tcen_h = np.array(Tcen_h)
Pnoz_h = np.array(Pnoz_h)
Tnoz_h = np.array(Tnoz_h)

# Parse cooling data
cooling_data = """
0 0 432.4 0 451
1 0 425.7 0 435.1
2 0 418 0 421
3 0 413 0 413
4 0 403.8 0 399.6
5 0 396 0 389.1
6 0 389 0 380.1
7 0 381.8 0 372.1
8 0 374.3 0 364.1
9 0 366.5 0 355.7
14 0 329.3 0 318.6
19 0 298.6 0 289.8
24 0 272 0 265.3
29 0 248 0 243.6
49 0 207.9 0 205
54 0 192.1 0 190
59 0 178.3 0 176.8
64 0 166.7 0 165.8
69 0 155 0 154.6
74 0 144.8 0 144.8
"""
lines = cooling_data.strip().split('\n')
cooling_times = []
Pcen_c = []
Tcen_c = []
Pnoz_c = []
Tnoz_c = []
for line in lines:
    parts = line.split()
    cooling_times.append(float(parts[0]))
    Pcen_c.append(float(parts[1]))
    Tcen_c.append(float(parts[2]))
    Pnoz_c.append(float(parts[3]))
    Tnoz_c.append(float(parts[4]))

cooling_times = np.array(cooling_times)
Pcen_c = np.array(Pcen_c)
Tcen_c = np.array(Tcen_c)
Pnoz_c = np.array(Pnoz_c)
Tnoz_c = np.array(Tnoz_c)

# Analyze heating for central
print("Heating Cycle - Central Heater:")
segments_cen = find_constant_p_segments(heating_times, Pcen_h, Tcen_h, min_points=4)
for start, end in segments_cen:
    t_seg = heating_times[start:end+1]
    t_rel = t_seg - t_seg[0]
    T_seg = Tcen_h[start:end+1]
    P_val = Pcen_h[start]
    try:
        popt, _ = curve_fit(exp_approach, t_rel, T_seg, p0=[T_seg[-1] + 10, T_seg[0] - T_seg[-1], 20], maxfev=10000)
        T_ss, A, tau = popt
        print(f"Power: {P_val:.2f}, Time Constant tau: {tau:.2f} min, T_ss: {T_ss:.2f}")
    except Exception as e:
        print(f"Power: {P_val:.2f}, Fit failed: {e}")

# Analyze heating for nozzle
print("\nHeating Cycle - Nozzle Heater:")
segments_noz = find_constant_p_segments(heating_times, Pnoz_h, Tnoz_h, min_points=4)
for start, end in segments_noz:
    t_seg = heating_times[start:end+1]
    t_rel = t_seg - t_seg[0]
    T_seg = Tnoz_h[start:end+1]
    P_val = Pnoz_h[start]
    try:
        popt, _ = curve_fit(exp_approach, t_rel, T_seg, p0=[T_seg[-1] + 10, T_seg[0] - T_seg[-1], 20], maxfev=10000)
        T_ss, A, tau = popt
        print(f"Power: {P_val:.2f}, Time Constant tau: {tau:.2f} min, T_ss: {T_ss:.2f}")
    except Exception as e:
        print(f"Power: {P_val:.2f}, Fit failed: {e}")

# Analyze cooling for central (whole dataset as one segment)
print("\nCooling Cycle - Central Heater:")
t_rel_cen_c = cooling_times - cooling_times[0]
try:
    popt, _ = curve_fit(exp_decay, t_rel_cen_c, Tcen_c, p0=[30, Tcen_c[0] - 30, 20], maxfev=10000)
    T_amb, A, tau = popt
    print(f"Time Constant tau: {tau:.2f} min, T_amb: {T_amb:.2f}")
except Exception as e:
    print(f"Fit failed: {e}")

# Analyze cooling for nozzle
print("\nCooling Cycle - Nozzle Heater:")
t_rel_noz_c = cooling_times - cooling_times[0]
try:
    popt, _ = curve_fit(exp_decay, t_rel_noz_c, Tnoz_c, p0=[30, Tnoz_c[0] - 30, 20], maxfev=10000)
    T_amb, A, tau = popt
    print(f"Time Constant tau: {tau:.2f} min, T_amb: {T_amb:.2f}")
except Exception as e:
    print(f"Fit failed: {e}")

# Plot heating cycle
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Time (min)')
ax1.set_ylabel('Temperature (°C)')
ax1.plot(heating_times, Tcen_h, label='T Central', color='blue')
ax1.plot(heating_times, Tnoz_h, label='T Nozzle', color='green')
ax1.tick_params(axis='y')
fig.suptitle('Heating Cycle')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.show()

# Plot cooling cycle
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Time (min)')
ax1.set_ylabel('Temperature (°C)')
ax1.plot(cooling_times, Tcen_c, label='T Central', color='blue')
ax1.plot(cooling_times, Tnoz_c, label='T Nozzle', color='green')
ax1.tick_params(axis='y')

fig.suptitle('Cooling Cycle')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.show()