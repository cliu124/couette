from time import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.io import savemat
import matplotlib.pyplot as plt

# whether to export data to a mat file
savedata = False

# Constants
Pr = 0.72  # value for air
gamma = 1.4  # value for air
C = 0.5  # given
M_r = 2.0  # Mach number

f1 = (gamma - 1) * M_r * M_r

T_r = 1 + f1 / 2 * Pr  # Recovery temperature

# Set up spatial grid
Ny = 501
y = np.linspace(0, 1, Ny)

# Set up time grid
t_span = (0, .5)  # Start and end times, default (0, 0.5)
Nt = 1001
t_eval = np.linspace(*t_span, Nt)  # Times at which to store the solution
t_diff = t_span[1] - t_span[0]

def solve():
    """Solves steady state ODE system using solve_ivp and root_scalar"""
    
    def ode_system(X, tau):
        """Define the system of ODEs and returns spatial derivatives"""
        U0, T = X
        
        # Viscosity
        etaRecip = (np.maximum(T, 1e-10) + C) / (np.maximum(T, 1e-10) ** (3 / 2) * (1 + C))  # equation 14
        
        # Derivatives
        dU0_dy = tau * etaRecip  # equation 10
        dT_dy = -(Pr * etaRecip) * ((gamma - 1) * M_r * M_r * tau * U0)  # equation 11
        
        return [dU0_dy, dT_dy]
    
    def shoot(tau):
        """Shooting method, returns boundary condition residual"""
        y_span = (0, 1)
        X0 = [0, T_r]  # Initial conditions: U0(0) = 0, T(0) = Tr
        
        sol = solve_ivp(lambda y, X: ode_system(X, tau), y_span, X0, dense_output=True)
        
        return sol.sol(1)[0] - 1

    tau_guess = 1 + M_r / 2

    root_result = root_scalar(lambda tau: shoot(tau), 
                              method='brentq', 
                              bracket=[0.1, gamma * (M_r + 1)], 
                              x0=tau_guess)
    
    if not root_result.converged:
        raise ValueError(f"Root finding failed to converge: {root_result.flag}")
    
    tau_solution = root_result.root
    print(f"Root found: tau = {tau_solution:.6f}, iterations: {root_result.iterations}")
    
    y_span = (0, 1)
    X0 = [0, T_r]
    sol = solve_ivp(lambda y, X: ode_system(X, tau_solution), y_span, X0, dense_output=True)
    
    y = np.linspace(0, 1, Ny)
    X = sol.sol(y)
    U0, T = X
    T += 1 - T[-1]  # enforce boundary condition of T0(1) = 1
    eta = T ** (3 / 2) * (1 + C) / (T + C)  # calculate viscosity
    
    return y, U0, T, eta

def viscosity(T):
    """Calculate viscosity based on temperature"""
    T = np.maximum(T, 0)
    return T ** (3 / 2) * (1 + C) / (T + C)

niter = 0

def pde_system(t, state, y):
    """Define the PDE system"""
    global niter

    N = len(y)
    U = state[:N]
    T = state[N:]
    
    # Calculate viscosity
    eta = viscosity(T)
    
    dUdy = np.gradient(U, y)
    dTdy = np.gradient(T, y)

    tau = eta * dUdy
    dUdt = np.gradient(tau, y)

    E = f1 * tau * U + eta / Pr * dTdy
    dTdt = np.gradient(E, y)

    # Enforce boundary conditions
    dUdt[0] = 0   # U(0, t) = 0
    dUdt[-1] = 0  # U(1, t) = 1
    dTdt[0] = 0   # T(0, t) = T_r
    dTdt[-1] = 0  # T(1, t) = 1
    
    print(f'\rSolution progress: {(100 * (t - t_span[0]) / (t_diff)):.2f}%, t={t:.10f}, niter={niter}', end='')
    niter += 1

    return np.concatenate((dUdt, dTdt))

# Initial U condition
U = np.zeros(Ny)

# Initial T condition
T0 = np.ones(Ny)
# T0 = np.linspace(T_r, 1, Ny)
# T0 = np.full(Ny, T_r)

# Boundary conditions
U[0]   = 0
U[-1]  = 1
T0[0]  = T_r
T0[-1] = 1

# Combine initial conditions
state0 = np.concatenate((U, T0))

t1 = time()

# Solve PDE
sol = solve_ivp(pde_system, t_span, state0, t_eval=t_eval, args=(y,), method='BDF')

if sol.status:
    raise Exception(sol.message)

print(f'\nDone\nniter: {niter}\nTime taken: {time() - t1}s')

# Extract U and T from solution
U_sol = sol.y[:Ny, :]
T_sol = sol.y[Ny:, :]

matData = {
    'y': np.flip(np.rot90([y], 1)),
    't': np.flip(np.rot90([t_eval], 1)),
    'U_sol': U_sol, #np.rot90(U_sol, 2),
    'T_sol': T_sol #np.rot90(T_sol, 2)
}

if savedata:
    path = f'export/couette_startup_M{M_r}_T{Nt}_Y{Ny}_{hex(round(time()))[2:]}.mat'
    savemat(path, matData)
    print(f'Saved to {path}')

# Plot results
plt.figure(figsize=(10, 8))

# Get steady state solution
print('Steady state calculation ', end='')
_, U_exact, T_exact, _ = solve()

from math import sqrt, floor

# time points to graph - show more detail at small t
t_points = (np.linspace(0, floor(sqrt(len(t_eval))), 50) ** 2 + 1).astype(int)[::2]
# t_points = list(range(0, len(t_eval), len(t_eval) // 50))
t_points[0] = 0

plt.subplot(2, 2, 1)
# Plot startup
for i in t_points:
    plt.plot(U_sol[:, i], y, label=f't = {t_eval[i]:.2f}')
# Plot steady state
plt.plot(U_exact, y, label=f't = inf', linestyle='dotted')
plt.xlabel('U')
plt.ylabel('y')
plt.title(f'Velocity (M = {M_r})')
# plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
# Plot startup
for i in t_points:
    plt.plot(T_sol[:, i], y, label=f't = {t_eval[i]:.2f}')
# Plot steady state
plt.plot(T_exact, y, label=f't = inf', linestyle='dotted')
plt.xlabel('T')
plt.ylabel('y')
plt.title(f'Temperature')
# plt.legend()
plt.grid(True)

plt.subplot(2,2,3)
plt.imshow(U_sol, aspect='auto', origin='lower', cmap = 'plasma', extent=[t_eval[0], t_eval[-1], y[0], y[-1]])
plt.title('Velocity')
plt.xlabel('Time')
plt.ylabel('y')
plt.colorbar(label='U')

plt.subplot(2,2,4)
plt.imshow(T_sol, aspect='auto', origin='lower', cmap='plasma', extent=[t_eval[0], t_eval[-1], y[0], y[-1]])
plt.title('Temperature')
plt.xlabel('Time')
plt.ylabel('y')
plt.colorbar(label='T')

plt.tight_layout()
plt.show()
