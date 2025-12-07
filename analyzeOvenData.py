import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- 1. Define the Gaussian Model Function ---
# This is kept outside the main function as it's a static definition
def gaussian(x, a, x0, sigma, c):
    """
    A 1D Gaussian function.
    
    Parameters:
    x (array): Independent variable (position)
    a (float): Amplitude (peak height above baseline)
    x0 (float): Center (peak position)
    sigma (float): Standard deviation (width)
    c (float): Baseline offset
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + c

def fit_atomic_beam_profile(x_data, y_data, plot=True, title=None):
    """
    Fits a 1D Gaussian function to atomic beam profile data.
    
    Parameters:
    x_data (array-like): The monitor position data.
    y_data (array-like): The thickness monitor reading data.
    plot (bool): If True, displays a plot of the data and the fit.
    title (str): Optional title for the plot.
    
    Returns:
    dict: A dictionary containing the fit parameters ('amplitude', 'center',
          'sigma', 'offset', 'fwhm') and the covariance matrix ('covariance').
          Returns None if the fit fails.
    """
    
    # Ensure data are numpy arrays for calculations
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    
    # --- 3. Provide Initial Guesses (Dynamically) ---
    try:
        guess_c = np.min(y_data)
        guess_a = np.max(y_data) - guess_c
        guess_x0 = x_data[np.argmax(y_data)]
        guess_sigma = (np.max(x_data) - np.min(x_data)) / 5.0
        
        initial_guesses = [guess_a, guess_x0, guess_sigma, guess_c]
    except Exception as e:
        print(f"Error generating initial guesses for {title or 'scan'}: {e}")
        return None

    # --- 4. Perform the Fit, Plot, and Return Results ---
    try:
        popt, pcov = curve_fit(gaussian, x_data, y_data, p0=initial_guesses)
        
        # Extract optimal parameters
        amplitude, center, sigma, offset = popt
        fwhm = 2.355 * np.abs(sigma) # Use abs(sigma) for robustness
        
        # --- 5. Plot the Results (if requested) ---
        if plot:
            plt.figure(figsize=(6.5, 4.875))
            
            # --- PLOTTING THE RAW POINTS (NORMALIZED) ---
            # Plot raw data shifted by the fitted center
            plt.scatter(x_data - center, y_data, label='Raw Data', c='blue', alpha=0.7, s=30)
            
            # --- PLOTTING THE FITTED GAUSSIAN (NORMALIZED) ---
            # Generate a smooth curve using the absolute x-axis
            x_fit_abs = np.linspace(np.min(x_data), np.max(x_data), 200)
            y_fit = gaussian(x_fit_abs, *popt)
            
            # Plot the fitted line, also shifted by the center
            plt.plot(x_fit_abs - center, y_fit, 'r-', label='Gaussian Fit', linewidth=2)
            
            # Add labels for the fit parameters
            # Note: The 'center' value is the *absolute* position in cm
            fit_info = (
                f"Amplitude (A): {amplitude:.3e}\n"
                f"Center (x₀):   {center:.3f} cm\n"
                f"Width (σ):     {sigma:.3f} cm\n"
                f"FWHM:          {fwhm:.3f} cm\n"
                f"Offset (C):    {offset:.3e}"
            )
            
            plt.text(0.05, 0.95, fit_info, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top', 
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
            
            if title:
                plt.title(title, fontsize=16)
            else:
                plt.title('Atomic Beam Profile Fit', fontsize=16)
            
            # --- UPDATE AXIS LABEL ---
            plt.xlabel('Position Relative to Peak (cm)', fontsize=14)
            plt.ylabel('Thickness Reading', fontsize=14)
            plt.legend()
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.show()

        # --- 6. Return the Results ---
        results = {
            'amplitude': amplitude,
            'center': center,
            'sigma': sigma,
            'offset': offset,
            'fwhm': fwhm,
            'fit_parameters': popt,
            'covariance': pcov,
            'x_data': x_data,
            'y_data': y_data
        }
        return results
    
    except RuntimeError as e:
        print(f"Error: The fit did not converge for {title or 'this scan'}. {e}")
        if plot:
            plt.figure(figsize=(8, 5))
            plt.scatter(x_data, y_data, label='Raw Data (Fit Failed)', c='red')
            plt.title(f"{title or 'Scan'} - FIT FAILED", fontsize=16)
            plt.xlabel('Monitor Position (cm)', fontsize=14)
            plt.ylabel('Thickness Reading', fontsize=14)
            plt.legend()
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.show()
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {title or 'scan'}: {e}")
        return None
    
def plot_scan_comparison(main_title, results1, legend1, results2, legend2, fwhm):
    """
    Plots two fitted Gaussian profiles on the same axes for comparison.

    Parameters:
    results1 (dict): The results dictionary from the first fit.
    title1 (str): The title/label for the first scan.
    results2 (dict): The results dictionary from the second fit.
    title2 (str): The title/label for the second scan.
    """
    plt.figure(figsize=(7, 4.5))

    for res, title, color in [(results1, legend1, 'steelblue'), (results2, legend2, 'orangered')]:
        # --- Calculate Integrated Area ---
        # The area under the Gaussian peak (above the offset) is A * sigma * sqrt(2*pi)
        area = res['amplitude'] * np.abs(res['sigma']) * np.sqrt(2 * np.pi)

        # --- Generate data for the plot ---
        # Use a wide enough range around the center to show the full curve
        x_range = np.linspace(res['center'] - 4 * res['sigma'], res['center'] + 4 * res['sigma'], 400)
        y_fit = gaussian(x_range, *res['fit_parameters'])

        # --- Create the label for the legend ---
        if fwhm:
            legend_label = (
                f"{title}, $\omega_0$ = {res['fwhm']:.1f}cm"
            )
        else:
            legend_label = (f"{title}")

        # f"  Atomic Flux: {area:.2e}"
        # --- Plot the curve, centered at 0 ---
        plt.plot(x_range - res['center'], y_fit, color=color, label=legend_label, linewidth=1.5)

        # --- Plot the raw data points, also centered ---
        plt.scatter(res['x_data'] - res['center'], res['y_data'], 
                    marker='s', color=color, alpha=0.6, s=40)

    plt.title(main_title, fontsize=14)
    plt.xlabel("Position Relative to Peak (cm)", fontsize=14)
    plt.ylabel("Atomic Flux (atom/sec)", fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.xlim(-1,1)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()




    
def main():
    # Scan 1: Exit Temperature = 480C, second window
    x_scan1 = [1.9, 1.95, 2, 2.05, 2.1, 2.125, 2.15, 2.175, 2.2, 2.225, 2.25, 2.275, 2.3, 2.325, 2.35, 2.375, 2.4, 2.425, 2.45, 2.475, 2.5, 2.525, 2.55, 2.575, 2.6, 2.625, 2.65, 2.675, 2.7, 2.75, 2.8, 2.85, 2.9]
    y_scan1 = [7.94e11, 9.52e11, 1.25e12, 1.98e12, 3.42e12, 4.10e12, 6.05e12, 1.05e13, 1.78e13, 2.42e13, 3.70e13, 4.25e13, 5.36e13, 6.15e13, 6.65e13, 7.69e13, 7.03e13, 6.84e13, 6.08e13, 4.92e13, 4.15e13, 2.84e13, 2.00e13, 1.29e13, 8.00e12, 5.12e12, 3.76e12, 3.31e12, 2.83e12, 2.00e12, 1.50e12, 1.16e12, 2.95e11]

    # Scan 2: Exit Temperature = 440C, second window
    x_scan2 = [1.9, 1.95, 2, 2.05, 2.1, 2.125, 2.15, 2.175, 2.2, 2.225, 2.25, 2.275, 2.3, 2.325, 2.35, 2.375, 2.4, 2.425, 2.45, 2.475, 2.5, 2.525, 2.55, 2.575, 2.6, 2.625, 2.65, 2.675, 2.7, 2.75, 2.8, 2.85, 2.9]
    y_scan2 = [2.79e11, 4.97e11, 5.92e11, 7.14e11, 7.89e11, 1.01e12, 1.44e12, 3.73e12, 9.13e12, 1.28e13, 1.85e13, 2.48e13, 2.85e13, 3.60e13, 3.91e13, 4.19e13, 4.26e13, 3.75e13, 3.22e13, 2.85e13, 2.14e13, 1.53e13, 1.11e13, 6.49e12, 3.62e12, 2.24e12, 1.87e12, 1.25e12, 8.71e11, 1.09e12, 4.08e11, 4.56e11, 4.97e11]

    # Scan 3: Exit Temperature = 440C, first window
    x_scan3 = [2.15, 2.2, 2.25, 2.3, 2.325, 2.35, 2.375, 2.4, 2.425, 2.45, 2.475, 2.5, 2.525, 2.55, 2.6, 2.7, 2.75, 2.8, 2.85]
    y_scan3 = [5.03e11, 9.18e11, 3.90e12, 2.75e13, 4.18e13, 5.32e13, 5.82e13, 5.04e13, 4.22e13, 2.86e13, 1.71e13, 8.01e12, 4.23e12, 2.92e12, 1.80e12, 1.90e12, 4.90e11, 4.49e11, 4.42e11]

    # --- CONVERT X DATA TO CM ---
    x_scan1_cm = np.asarray(x_scan1) * 2.54
    x_scan2_cm = np.asarray(x_scan2) * 2.54
    x_scan3_cm = np.asarray(x_scan3) * 2.54
    
    # --- Update all_scans to use the _cm data ---
    all_scans = [
        ("Scan 1: 480C, 2nd window", x_scan1_cm, y_scan1),
        ("Scan 2: 440C, 2nd window", x_scan2_cm, y_scan2),
        ("Scan 3: 440C, 1st window", x_scan3_cm, y_scan3)
    ]

    all_results = []
    for title, x_data, y_data in all_scans:
        print(f"--- Fitting {title} ---")
        
        results = fit_atomic_beam_profile(x_data, y_data, plot=False, title=title)
        
        if results: #
            print(f"Successfully fit {title}.")
            print(f"  Center (x₀): {results['center']:.4f}")
            print(f"  FWHM:        {results['fwhm']:.4f}")
            all_results.append(results)
        else:
            print(f"Fit failed for {title}.")
        
        print("-" * (17 + len(title))) 
    
    # 3. You can now access all results programmatically
    print("\n--- Summary of All Fits ---")
    for i, res in enumerate(all_results):
        print(f"Scan {i+1} Center: {res['center']:.4f} \t FWHM: {res['fwhm']:.4f}")

    # --- Add Comparison Plots ---
    if len(all_results) == 3:
        # Compare Scan 2 and Scan 3
        plot_scan_comparison("Atomic Flux vs. Longitudinal Deflection (T = 440C)", all_results[2], "z = 0cm",
                             all_results[1], "z = 9cm", True)

        # Compare Scan 1 and Scan 2
        plot_scan_comparison("Atomic Flux vs. Oven Temperature (at Cavity)", all_results[1], "T = 440C",
                             all_results[0], "T = 480C", False)

if __name__ == "__main__":
    main()