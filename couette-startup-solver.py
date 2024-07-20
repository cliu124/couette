from time import time

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
Pr = 0.72  # value for air
gamma = 1.4  # value for air
C = 0.5  # given
M_r = 2.0  # Mach number (you can change this)
T_r = 1 + (gamma - 1) / 2 * Pr * M_r**2  # Recovery temperature

f1 = (gamma - 1) * M_r*M_r

def temperature(U):
    """Calculate temperature based on velocity"""
    return T_r * (1 + (1 - 1/T_r) * U - (1 - 1/T_r) * U**2)

def viscosity(T):
    """Calculate viscosity based on temperature"""
    return np.maximum(T, 0) ** (3 / 2) * (1 + C) / (np.maximum(T, 0) + C)

def pde_system(t, state, y):
    """Define the PDE system"""
    N = len(y)
    U = state[:N]
    T = state[N:]
    
    y = np.array(y)
    
    dUdt = np.zeros(N)
    dTdt = np.zeros(N)
    
    # Calculate viscosity
    eta = viscosity(T)
    
    dy = y[1] - y[0]
    dy2 = dy * dy
    
    for i in range(1, N-1):
        dU1 = U[i+1] - U[i]
        dU2 = U[i] - U[i-1]
    
        # Finite difference for momentum equation
        dUdt[i] = (eta[i+1] * (dU1) - eta[i-1] * (dU2)) / dy2 / T[i]
    
        # Finite difference for temperature equation
        dTdt[i] = (f1 * ((eta[i+1] * (dU1 * dU2)) - (eta[i-1] * (dU1 * dU2))) / dy + ((eta[i+1] * (T[i+1] - T[i])) - (eta[i-1] * (T[i] - T[i-1]))) / Pr) / dy2
    
    # Enforce boundary conditions
    dUdt[0] = 0  # U(0, t) = 0
    dUdt[-1] = 0  # U(1, t) = 1
    dTdt[0] = 0  # T(0, t) = T_r
    dTdt[-1] = 0  # T(1, t) = 1
    
    return np.concatenate((dUdt, dTdt))

# Set up spatial grid
Ny = 101
y = np.linspace(0, 1, Ny)

# Set up time grid
t_span = (0, 0.5)  # Start and end times
Nt = 501
t_eval = np.linspace(*t_span, Nt)  # Times at which to store the solution

# Initial condition
U0 = np.zeros(Ny)
U0[-1] = 1  # Enforce U(1, t) = 1

T0 = np.ones(Ny)
T0[0] = T_r  # Enforce T(0, t) = T_r

# Combine initial conditions
state0 = np.concatenate((U0, T0))

# Solve PDE
t1 = time()
sol = solve_ivp(pde_system, t_span, state0, t_eval=t_eval, args=(y,), method='RK45')
print(time() - t1)

print('Done')

# Extract U and T from solution
U_sol = sol.y[:Ny, :]
T_sol = temperature(U_sol) #sol.y[Ny:, :]

# Plot results
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
for i in range(0, len(t_eval), len(t_eval) // 50):
    plt.plot(U_sol[:, i], y, label=f't = {t_eval[i]:.2f}')
plt.xlabel('U')
plt.ylabel('y')
plt.title(f'Startup Compressible Couette Flow (M = {M_r})')
# plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for i in range(0, len(t_eval), len(t_eval) // 50):
    plt.plot(T_sol[:, i], y, label=f't = {t_eval[i]:.2f}')
plt.xlabel('T')
plt.ylabel('y')
plt.title(f'Temperature Distribution (M = {M_r})')
# plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(U_sol, aspect='auto', origin='lower', extent=[t_eval[0], t_eval[-1], y[0], y[-1]])
ax1.set_title('Velocity')
ax1.set_xlabel('Time')
ax1.set_ylabel('y')
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(T_sol, aspect='auto', origin='lower', cmap='plasma', extent=[t_eval[0], t_eval[-1], y[0], y[-1]])
ax2.set_title('Temperature')
ax2.set_xlabel('Time')
ax2.set_ylabel('y')
plt.colorbar(im2, ax=ax2)
plt.show()