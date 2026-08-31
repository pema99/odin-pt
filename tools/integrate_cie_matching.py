# Integrals of the matching functions in shaders/spectrum.slang over the visible range
import numpy as np

MIN_VISIBLE_WAVELENGTH = 380.0
MAX_VISIBLE_WAVELENGTH = 780.0

def cie_x_matching(wavelengths):
    t1 = (wavelengths-442.0)*np.where(wavelengths<442.0, 0.0624, 0.0374)
    t2 = (wavelengths-599.8)*np.where(wavelengths<599.8, 0.0264, 0.0323)
    t3 = (wavelengths-501.1)*np.where(wavelengths<501.1, 0.0490, 0.0382)
    return 0.362*np.exp(-0.5*t1*t1) + 1.056*np.exp(-0.5*t2*t2) \
                                    - 0.065*np.exp(-0.5*t3*t3)

def cie_y_matching(wavelengths):
    t1 = (wavelengths-568.8)*np.where(wavelengths<568.8, 0.0213, 0.0247)
    t2 = (wavelengths-530.9)*np.where(wavelengths<530.9, 0.0613, 0.0322)
    return 0.821*np.exp(-0.5*t1*t1) + 0.286*np.exp(-0.5*t2*t2)

def cie_z_matching(wavelengths):
    t1 = (wavelengths-437.0)*np.where(wavelengths<437.0, 0.0845, 0.0278)
    t2 = (wavelengths-459.0)*np.where(wavelengths<459.0, 0.0385, 0.0725)
    return 1.217*np.exp(-0.5*t1*t1) + 0.681*np.exp(-0.5*t2*t2)

wavelengths = np.linspace(MIN_VISIBLE_WAVELENGTH, MAX_VISIBLE_WAVELENGTH, 400001)
integrals = [np.trapz(f(wavelengths), wavelengths)
             for f in (cie_x_matching, cie_y_matching, cie_z_matching)]
print("static const float3 CIE_MATCHING_INTEGRAL = float3(%.6f, %.6f, %.6f);" % tuple(integrals))
