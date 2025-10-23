import requests
import yaml

# Install necessary libraries if you haven't already:
# pip install requests pyyaml

def get_rims_scheme(element_symbol):
    """
    Fetches and parses the RIMS scheme for a given element.

    Args:
        element_symbol (str): The symbol of the element (e.g., 'Sr', 'U').

    Returns:
        dict: A dictionary containing the parsed YAML data, or None if not found.
    """
    # Construct the URL for the raw YAML file on GitHub
    url = f"https://raw.githubusercontent.com/RIMS-Code/rims-code.github.io/master/_schemes/{element_symbol.lower()}.yml"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (404, 500, etc.)

        # Parse the YAML content
        scheme_data = yaml.safe_load(response.text)
        return scheme_data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {element_symbol}: {e}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML for {element_symbol}: {e}")
        return None

# --- Example Usage ---
element = "Sr"
sr_data = get_rims_scheme(element)

if sr_data:
    # Print the full data structure to inspect it
    # import json
    # print(json.dumps(sr_data, indent=2))

    # Access specific information from the parsed data
    print(f"Successfully retrieved scheme for: {sr_data.get('element')}")
    print(f"Mass Number: {sr_data.get('mass_number')}")

    # Loop through the defined schemes
    for i, scheme in enumerate(sr_data.get('schemes', []), 1):
        print(f"\n--- Scheme {i} ---")
        print(f"  Reference: {scheme.get('reference')}")
        print(f"  Note: {scheme.get('note')}")

        # Loop through the lasers in the scheme
        for laser in scheme.get('lasers', []):
            print(f"  - Laser:")
            print(f"    Wavelength (nm): {laser.get('wavelength')}")
            print(f"    Transition: {laser.get('transition')}")
            print(f"    Level (cm-1): {laser.get('level')}")