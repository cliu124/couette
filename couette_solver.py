import threading
# import multiprocessing
# import math
import time

import numpy as np
from scipy.integrate import solve_bvp
from scipy.io import savemat
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mplcursors import cursor


def cubic_spline_interpolate(data, target_length):
    x_original = np.linspace(0, 1, len(data))
    x_new = np.linspace(0, 1, target_length)
    cs = CubicSpline(x_original, data)
    data_interpolated = cs(x_new)
    return data_interpolated

# NUM_CPU = multiprocessing.cpu_count() - 2

# Constants
Pr = 0.72  # value for air
gamma = 1.4  # value for air
C = 0.5  # given
tau_guess = 2  # Initial guess 1.5
M = np.linspace(0, 5, 11) # [0, 0.2, 0.5, 1, 2, 5]   # Mach numbers to test

lock = threading.Lock()

def ode_system(y, inputs, tau, C, gamma, M_r):
    """Define the system of ODEs, returns derivatives based on inputs"""
    U0, T = inputs

    # Viscosity
    eta = T ** (3 / 2) * (1 + C) / (T + C)  # equation 14

    # Derivatives
    dU0_dy = tau / eta  # equation 10
    dT_dy = -(Pr / eta) * ((gamma - 1) * M_r**2 * tau * U0)  # equation 11

    return np.vstack((dU0_dy, dT_dy))


# Define boundary conditions
def bc(Xa, Xb, tau, C, gamma, M_r):
    """Returns residuals of boundary conditions"""
    U0a, Ta = Xa
    U0b, Tb = Xb
    return np.array(
        [U0a, U0b - 1]
    )  # Returns the residuals of the boundary conditions U0(0) = 0, U0(1) = 1

def solve(tau_guess, C, gamma, M_r):
    """Solves ODE system using solve_bvp"""
    # tau_guess = 1 + M_r / 2
    Tr = 1 + (gamma - 1) / 2 * M_r**2  # Recovery temperature
    y = np.linspace(0, 1, 1001)
    U0_guess = y  # Linear guess for U0
    T_guess = np.full_like(y, Tr)  # Constant guess for T
    X_guess = np.vstack((U0_guess, T_guess))
    sol = None
    l = 0
    
    # while l < 3000:
    sol = solve_bvp(
        lambda y, X: ode_system(y, X, tau_guess, C, gamma, M_r),
        lambda Xa, Xb: bc(Xa, Xb, tau_guess, C, gamma, M_r),
        y,
        X_guess,
        max_nodes=100000,
        tol=1e-10,
        # verbose=2,
    )
    # l = len(sol.x)
        # tau_guess *= 1.1

    # print(sol)

    if not sol.success:
        raise ValueError(sol.message)  # ValueError("Failed to find a solution")

    y = sol.x
    U = sol.y[0]
    T = sol.y[1]
    T += 1 - T[-1]  # enforce boundary condition of T0(1) = 1
    
    
    if len(sol.x) != 3001:
        print(f" Length: {len(y)}, niter: {sol.niter}")
        y = cubic_spline_interpolate(y, 3001)
        U = cubic_spline_interpolate(U, 3001)
        T = cubic_spline_interpolate(T, 3001)

    eta = T ** (3 / 2) * (1 + C) / (T + C)  # calculate viscosity

    # step = len(y)/3001
    # y1=[y[math.floor(step * i)] for i in range(3001)]
    # U1=[U[math.floor(step * i)] for i in range(3001)]
    # T1=[T[math.floor(step * i)] for i in range(3001)]
    # eta1=[eta[math.floor(step * i)] for i in range(3001)]
    # with lock:
    # data[str(M_r)] = [sol.x, sol.y[0], T, eta, T]  # y, U0, T, eta, xi
    return y, U, T, eta, T #y1, U1, T1, eta1, T1


plt.figure(figsize=(8, 7))

styles = ["solid", "dashed", "dotted"]

data = {"y": [], "M_r": [M], "U0": [], "T": [], "eta": [], "xi": []}

"""
threads = []

for M_r in M:
    threads.append(threading.Thread(target=solve, args=(tau_guess, C, gamma, M_r)))

for i in range(math.ceil(len(M) / NUM_CPU)):
    start = i * NUM_CPU
    end = start + NUM_CPU
    m = M[start : end]
    print(f'Calculating mach numbers {m}')
    batch = threads[start : end]
    [thread.start() for thread in batch]

    for thread in batch:
        thread.join()
"""
y_all = []
u0_all = []
# Plot curves for each mach number
for i, M_r in enumerate(M):
    print(f'\rCalculating mach {round(M_r*100)/100}', end='')
    y, U0, T, eta, xi = solve(tau_guess, C, gamma, M_r)  # solve
    # y, U0, T, eta, xi = data[str(M_r)]

    if i == 0:
        data["y"].append(list(y))
        y_all.append(y)
    u0_all.append(U0)
    data["U0"].append(list(U0))
    data["T"].append(list(T))
    data["eta"].append(list(eta))
    data["xi"].append(list(xi))

    # Velocity
    plt.subplot(2, 2, 1)
    # plt.subplot(2,1,1)
    plt.axis((0, 1, 0, 1))
    plt.plot(U0, y, label=f"Mr = {M_r}", linestyle=styles[i % len(styles)])
    plt.ylabel("y")
    plt.xlabel("U0")
    plt.title("Velocity")
    plt.legend()

    # Temperature
    plt.subplot(2, 2, 2)
    plt.axis((1, max(T) * 1.1, 0, 1))  # [1, 1.6]
    plt.plot(T, y, label=f"Mr = {M_r}", linestyle=styles[i % 3])
    plt.ylabel("y")
    plt.xlabel("T0")
    plt.title("Temperature")
    # plt.legend()

    # Specific volume (xi)
    plt.subplot(2, 2, 3)
    plt.axis((1, max(xi) * 1.1, 0, 1))  # [1, 1.6]
    plt.plot(T, y, label=f"Mr = {M_r}", linestyle=styles[i % 3])
    plt.ylabel("y")
    plt.xlabel("Xi0")
    plt.title("Specific volume")
    # plt.legend()

    # Viscosity coefficient (eta)
    plt.subplot(2, 2, 4)
    plt.axis((0.9, max(eta) * 1.1, 0, 1))  # [0.1, 1.5]
    plt.plot(eta, y, label=f"Mr = {M_r}", linestyle=styles[i % 3])
    plt.ylabel("y")
    plt.xlabel("Eta0")
    plt.title("Viscosity coefficient")
    # plt.legend()

savemat(f"export/couette_data_{hex(round(time.time()))[2:]}.mat", data)

cursor(hover=True)
plt.tight_layout()
plt.show()

y=data['y']
U0=data['U0']
T=data['T']
M_r=data['M_r']

plt.subplot(1,2,1)
plt.imshow(
    np.rot90(U0),
    aspect="auto",
    origin="upper",
    extent=[min(M_r[0]), max(M_r[0]), 0, 1],
    cmap="plasma",
    # norm="log"
)
# cursor(hover=True)
plt.colorbar(label="U")
plt.title("Velocity")
plt.xlabel("M_r")
plt.ylabel("y")


plt.subplot(1,2,2)
plt.imshow(
    np.rot90(T),
    aspect="auto",
    origin="upper",
    extent=[min(M_r[0]), max(M_r[0]), 0, 1],
    cmap="plasma",
    # norm="log"
)
# cursor(hover=True)
plt.colorbar(label="T")
plt.title("Temperature")
plt.xlabel("M_r")
plt.ylabel("y")
plt.show()
