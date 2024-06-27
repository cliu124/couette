import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Constants
Pr = 0.72         # value for air
gamma = 1.4       # value for air
C = 0.5           # given
tau_guess = 1     # Initial guess
M = [0.5, 1, 2]   # Mach numbers to test

# Define system of equations
def ode_system(y, state, tau, C, gamma, M_r):
  U0, T = state
  
  # Viscosity
  eta = T ** (3/2) * (1 + C) / (T + C)
  
  # Derivatives
  dU0_dy = tau / eta
  dT_dy = -(Pr / eta) * ((gamma - 1) * M_r**2 * tau * U0)
  
  return np.vstack((dU0_dy, dT_dy))

# Define boundary conditions
def bc(Xa, Xb, tau, C, gamma, M_r):
  U0a, Ta = Xa
  U0b, Tb = Xb
  return np.array([U0a, U0b - 1])   # Returns the residuals of the boundary conditions U0(0) = 0, U0(1) = 1

def solve(tau_guess, C, gamma, M_r, Tr):
  y = np.linspace(0, 1, 1000)
  U0_guess = y                      # Linear guess for U0
  T_guess = np.full_like(y, Tr)     # Constant guess for T
  X_guess = np.vstack((U0_guess, T_guess))    
  
  sol = solve_bvp(lambda y, X: ode_system(y, X, tau_guess, C, gamma, M_r),
                  lambda Xa, Xb: bc(Xa, Xb, tau_guess, C, gamma, M_r),
                  y, X_guess)
  
  if not sol.success:
    raise ValueError("Failed to find a solution")
  
  T = sol.y[1]
  T += (1 - T[-1])    # enforce boundary condition of T0(1) = 1
  eta = T ** (3/2) * (1 + C) / (T + C)

  return sol.x, sol.y[0], T, eta  # y, U0, T

for M_r in M:
  Tr = 1 + (gamma - 1) / 2 * M_r**2  # Recovery temperature  
  y, U0, T, eta = solve(tau_guess, C, gamma, M_r, Tr)    # solve

  plt.subplot(2, 2, 1)
  plt.plot(U0, y, label=f'M0 = {M_r}')
  plt.ylabel('y')
  plt.xlabel('U0')
  plt.title('Velocity')
  plt.legend()

  plt.subplot(2, 2, 2)
  plt.plot(T, y, label=f'M0 = {M_r}')
  plt.ylabel('y')
  plt.xlabel('T0')
  plt.title('Temperature')
  plt.legend()
  
  plt.subplot(2, 2, 4)
  plt.plot(eta, y, label=f'M0 = {M_r}')
  plt.ylabel('y')
  plt.xlabel('Eta0')
  plt.title('Viscosity coefficient')
  plt.legend()

plt.tight_layout()
plt.show()