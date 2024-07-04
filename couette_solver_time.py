import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import threading

# Constants
Pr = 0.72  # value for air
gamma = 1.4  # value for air
C = 0.5  # given
tau_guess = 1  # Initial guess
M = [0, 0.2, 0.5, 1, 2]  # Mach numbers to test


# Time evolution using finite difference method
def time_evolution(U, T, tau, dy, dt, nt, ny, C, gamma, M_r, Pr):
    U_history = np.zeros((nt, ny))
    T_history = np.zeros((nt, ny))
    U_history[0, :] = U
    T_history[0, :] = T

    for t in range(1, nt):
        eta = T ** (3 / 2) * (1 + C) / (T + C)
        U_new = np.copy(U)
        T_new = np.copy(T)

        for j in range(1, ny - 1):
            U_new[j] = U[j] + dt * (eta[j] * (U[j + 1] - 2 * U[j] + U[j - 1]) / dy**2)
            T_new[j] = (
                T[j]
                + dt
                * ((gamma - 1) * M_r**2 * tau * U[j] + eta[j] / Pr)
                * (T[j + 1] - 2 * T[j] + T[j - 1])
                / dy**2
            )

        # Apply boundary conditions
        U_new[0], U_new[-1] = 0, 1
        T_new[0], T_new[-1] = T[1], 1

        U, T = U_new, T_new
        U_history[t, :] = U
        T_history[t, :] = T

    return np.rot90(U_history), np.rot90(T_history)


# Initial conditions
ny = 100
dy = 1 / (ny - 1)
dt = 1e-5
nt = 5000
y = np.linspace(0, 1, ny)

data = {}


def calculate(M_r):
    """calculate based on mach number"""
    U = np.copy(y)  # Initial condition for U
    T = np.full(ny, 1.0)  # Initial condition for T
    Tr = 1 + (gamma - 1) / 2 * M_r**2  # Recovery temperature
    T = np.full(ny, Tr)  # Initial condition for T based on Tr
    tau = tau_guess
    with lock:
        data[str(M_r)] = time_evolution(U, T, tau, dy, dt, nt, ny, C, gamma, M_r, Pr)


lock = threading.Lock()

threads = []

for M_r in M:
    threads.append(threading.Thread(target=calculate, args=(M_r,)))
    threads[-1].start()

for thread in threads:
    thread.join()

plt.figure(figsize=(16, 8))

# Plot curves for each Mach number
for i, M_r in enumerate(M):
    U_history, T_history = data[str(M_r)]
    # Plot the results for U
    plt.subplot(2, len(M), i + 1)
    plt.imshow(
        U_history,
        aspect="auto",
        origin="upper",
        extent=[0, 1, 0, nt * dt],
        cmap="viridis",
    )
    plt.colorbar(label="U")
    plt.title(f"Velocity (Mr={M_r})")
    plt.xlabel("t")
    plt.ylabel("y")

    # Plot the results for T
    plt.subplot(2, len(M), len(M) + i + 1)
    plt.imshow(
        T_history,
        aspect="auto",
        origin="upper",
        extent=[0, 1, 0, nt * dt],
        cmap="inferno",
    )  # , vmin=1, vmax=1.5)
    plt.colorbar(label="T")
    plt.title(f"Temperature (Mr={M_r})")
    plt.xlabel("t")
    plt.ylabel("y")

savemat("export/couette_time.mat", data)

plt.tight_layout()
plt.show()
