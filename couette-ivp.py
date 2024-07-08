import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import savemat
import matplotlib.pyplot as plt
from mplcursors import cursor

# Constants
Pr = 0.72  # value for air
gamma = 1.4  # value for air
C = 0.5  # given
M = np.linspace(0, 5, 101)  # Mach numbers to test

def ode_system(y, X, tau, C, gamma, M_r):
    """Define the system of ODEs"""
    U0, T = X
    
    # Viscosity
    eta = np.maximum(T, 1e-10) ** (3 / 2) * (1 + C) / (np.maximum(T, 1e-10) + C)  # equation 14
    
    # Derivatives
    dU0_dy = tau / eta  # equation 10
    dT_dy = -(Pr / eta) * ((gamma - 1) * M_r**2 * tau * U0)  # equation 11
    
    return [dU0_dy, dT_dy]

def shoot(tau, C, gamma, M_r):
    """Shooting method"""
    Tr = 1 + (gamma - 1) / 2 * M_r**2  # Recovery temperature
    y_span = (0, 1)
    X0 = [0, Tr]  # Initial conditions: U0(0) = 0, T(0) = Tr
    
    sol = solve_ivp(lambda y, X: ode_system(y, X, tau, C, gamma, M_r), y_span, X0, dense_output=True)
    
    return sol.sol(1)[0] - 1  # Return the difference from the target value U0(1) = 1

def secant_method(f, x0, x1, tol=1e-8, max_iter=100):
    """Secant method for finding the root of f(x)"""
    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        if abs(fx1) < tol:
            return x1, i
        if fx0 == fx1:
            raise ValueError("Secant method failed: division by zero")
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x_new
    raise ValueError(f"Secant method failed to converge after {max_iter} iterations")

def bisection_method(f, a, b, tol=1e-8, max_iter=100):
    """Bisection method for finding the root of f(x)"""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Function values at a and b must have opposite signs")
    
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        if abs(fc) < tol:
            return c, i
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    
    raise ValueError(f"Bisection method failed to converge after {max_iter} iterations")

def solve(C, gamma, M_r):
    """Solves ODE system using solve_ivp and secant method with fallback to bisection"""
    tau_guess_1 = 1 + M_r / 2
    tau_guess_2 = tau_guess_1 * 1.1
    
    try:
        tau_solution, iterations = secant_method(lambda tau: shoot(tau, C, gamma, M_r), tau_guess_1, tau_guess_2)
        print(f"Secant method converged in {iterations} iterations")
    except ValueError as err:
        print(f"Secant method failed: {err}")
        try:
            tau_solution, iterations = bisection_method(lambda tau: shoot(tau, C, gamma, M_r), 0.1, 10)
            print(f"Bisection method converged in {iterations} iterations")
        except ValueError as err:
            print(f"Bisection method failed: {err}")
            raise ValueError("Both secant and bisection methods failed to converge")
    
    y_span = (0, 1)
    X0 = [0, 1 + (gamma - 1) / 2 * M_r**2]
    sol = solve_ivp(lambda y, X: ode_system(y, X, tau_solution, C, gamma, M_r), y_span, X0, dense_output=True)
    
    y = np.linspace(0, 1, 3001)
    X = sol.sol(y)
    U0, T = X
    T += 1 - T[-1]  # enforce boundary condition of T0(1) = 1
    eta = T ** (3 / 2) * (1 + C) / (T + C)  # calculate viscosity
    
    return y, U0, T, eta, T

# Main calculation and plotting code
plt.figure(figsize=(8, 7))
styles = ["solid", "dashed", "dotted"]
data = {"y": [], "M_r": [M], "U0": [], "T": [], "eta": [], "xi": []}

for i, M_r in enumerate(M):
    print(f"\nCalculating mach {round(M_r*100)/100}")
    try:
        y, U0, T, eta, xi = solve(C, gamma, M_r)
    except ValueError as e:
        print(f"Error for M_r = {M_r}: {e}")
        continue
    
    if i == 0:
        data["y"].append(list(y))
    data["U0"].append(list(U0))
    data["T"].append(list(T))
    data["eta"].append(list(eta))
    data["xi"].append(list(xi))

    # Velocity
    plt.subplot(2, 2, 1)
    plt.axis((0, 1, 0, 1))
    plt.plot(U0, y, label=f"Mr = {M_r}", linestyle=styles[i % len(styles)])
    plt.ylabel("y")
    plt.xlabel("U0")
    plt.title("Velocity")
    plt.legend()
    
    # Temperature
    plt.subplot(2, 2, 2)
    plt.axis((1, max(T) * 1.1, 0, 1))
    plt.plot(T, y, label=f"Mr = {M_r}", linestyle=styles[i % 3])
    plt.ylabel("y")
    plt.xlabel("T0")
    plt.title("Temperature")
    
    # Specific volume (xi)
    plt.subplot(2, 2, 3)
    plt.axis((1, max(xi) * 1.1, 0, 1))
    plt.plot(T, y, label=f"Mr = {M_r}", linestyle=styles[i % 3])
    plt.ylabel("y")
    plt.xlabel("Xi0")
    plt.title("Specific volume")
    
    # Viscosity coefficient (eta)
    plt.subplot(2, 2, 4)
    plt.axis((0.9, max(eta) * 1.1, 0, 1))
    plt.plot(eta, y, label=f"Mr = {M_r}", linestyle=styles[i % 3])
    plt.ylabel("y")
    plt.xlabel("Eta0")
    plt.title("Viscosity coefficient")

savemat(f"export/ivp_couette_data_{hex(round(time.time()))[2:]}.mat", data)

cursor(hover=True)
plt.tight_layout()
plt.show()

# Additional plots
y = data["y"][0]
U0 = data["U0"]
T = data["T"]
M_r = data["M_r"][0]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(
    np.rot90(U0),
    aspect="auto",
    origin="upper",
    extent=[min(M_r), max(M_r), 0, 1],
    cmap="plasma",
)
plt.colorbar(label="U")
plt.title("Velocity")
plt.xlabel("M_r")
plt.ylabel("y")

plt.subplot(1, 2, 2)
plt.imshow(
    np.rot90(T),
    aspect="auto",
    origin="upper",
    extent=[min(M_r), max(M_r), 0, 1],
    cmap="plasma",
)
plt.colorbar(label="T")
plt.title("Temperature")
plt.xlabel("M_r")
plt.ylabel("y")

plt.tight_layout()
plt.show()