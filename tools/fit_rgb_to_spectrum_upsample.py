# Based on Spectral Primary Decomposition for Rendering with sRGB Reflectance, Mallett and Yuksel,
# https://graphics.geometrian.com/research/spectral-primaries.html

# - Uses the "minimize maximum differences between consecutive wavelengths" objective from the paper.
# - Refit for Illuminant E as the white point (rather than D65)
# - Assuming Wyman's CIE matching function approximation (https://jcgt.org/published/0002/02/01/)

import numpy as np
from scipy.optimize import linprog, minimize

MIN_BASIS_WAVELENGTH = 380.0
MAX_BASIS_WAVELENGTH = 780.0
BASIS_WAVELENGTH_STEP = 5.0
PRIMARIES = [(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)]

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

wavelengths = np.arange(MIN_BASIS_WAVELENGTH,
                        MAX_BASIS_WAVELENGTH + 0.5*BASIS_WAVELENGTH_STEP,
                        BASIS_WAVELENGTH_STEP)
n = len(wavelengths)
steps = n - 1
cie = np.stack([cie_x_matching(wavelengths),
                cie_y_matching(wavelengths),
                cie_z_matching(wavelengths)], -1)

x = np.array([p[0] for p in PRIMARIES])
y = np.array([p[1] for p in PRIMARIES])
p = np.array([x/y, np.ones(3), (1-x-y)/y])
rgb_to_xyz = p * np.linalg.solve(p, cie.sum(0))

A_eq = np.zeros((9, 3*n+3))
b_eq = np.zeros(9)
for k in range(3):
    A_eq[3*k:3*k+3, k*n:(k+1)*n] = cie.T
    b_eq[3*k:3*k+3] = rgb_to_xyz[:, k]

A_ub = np.zeros((n + 6*steps, 3*n+3))
b_ub = np.zeros(n + 6*steps)
A_ub[:n, :3*n] = np.hstack([np.eye(n)]*3)
b_ub[:n] = 1.0
row = n
for k in range(3):
    for i in range(steps):
        for sign in (1.0, -1.0):
            A_ub[row, k*n+i+1] = sign
            A_ub[row, k*n+i] = -sign
            A_ub[row, 3*n+k] = -1.0
            row += 1

c = np.zeros(3*n+3)
c[3*n:] = 1.0
res = linprog(c, A_ub, b_ub, A_eq, b_eq,
              bounds=[(0, 1)]*(3*n) + [(0, None)]*3, method="highs")
basis = np.clip(res.x[:3*n].reshape(3, n).T, 0.0, 1.0)

print("public static const int RGB_TO_SPECTRUM_LUT_SIZE = %d;" % n)
print("")
print("public static const float3 RGB_TO_SPECTRUM_LUT[RGB_TO_SPECTRUM_LUT_SIZE] = {")
for i in range(0, n, 2):
    print("    " + " ".join("float3(%.6f, %.6f, %.6f)," % (r, g, b) for r, g, b in basis[i:i+2]))
print("};")


# Cheap gaussian apprxoimation
fine = np.linspace(MIN_BASIS_WAVELENGTH, MAX_BASIS_WAVELENGTH, 4001)
fine_cie = np.stack([cie_x_matching(fine), cie_y_matching(fine), cie_z_matching(fine)], -1)
xyz_to_rgb = np.linalg.inv(p * np.linalg.solve(p, fine_cie.sum(0)))

def gaussian_round_trip(params):
    mu, sigma = params[:3], params[3:]
    weights = np.exp(-((fine[:, None]-mu)/sigma)**2)
    weights = weights / weights.sum(1, keepdims=True)
    return xyz_to_rgb @ (weights[:, None, :]*fine_cie[:, :, None]).sum(0)

def gaussian_cost(params):
    if np.any(params[3:] < 5.0) or np.any(params[:3] < MIN_BASIS_WAVELENGTH) or np.any(params[:3] > MAX_BASIS_WAVELENGTH):
        return 1e3
    return np.abs(gaussian_round_trip(params) - np.eye(3)).max()

start = np.array([648.3, 516.5, 441.4, 26.6, 32.3, 49.2])
best, best_cost = start, gaussian_cost(start)
rng = np.random.default_rng(0)
for attempt in range(40):
    guess = start if attempt == 0 else start*(1 + rng.normal(0, 0.12, 6))
    result = minimize(gaussian_cost, guess, method="Nelder-Mead",
                      options=dict(maxiter=20000, maxfev=20000, xatol=1e-8, fatol=1e-12))
    if result.fun < best_cost:
        best, best_cost = result.x, result.fun

print("")
for name, mu, sigma in zip("rgb", best[:3], best[3:]):
    print("    float4 t%s = (wavelengths - %.4f) / %.4f;" % (name, mu, sigma))
