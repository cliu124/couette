import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Constants
Pr = 0.72         # value for air
gamma = 1.4       # value for air
C = 0.5           # given
tau_guess = 1.5     # Initial guess
M = [0.5, 1, 2]   # Mach numbers to test

# Define system of equations
def ode_system(y, inputs, tau, C, gamma, M_r):
  U0, T = inputs

  # Viscosity
  eta = T ** (3/2) * (1 + C) / (T + C)    # equation 14

  # Derivatives
  dU0_dy = tau / eta                                        # equation 10
  dT_dy = -(Pr / eta) * ((gamma - 1) * M_r**2 * tau * U0)   # equation 11
  
  return np.vstack((dU0_dy, dT_dy))

# Define boundary conditions
def bc(Xa, Xb, tau, C, gamma, M_r):
  U0a, Ta = Xa
  U0b, Tb = Xb
  return np.array([U0a, U0b - 1])   # Returns the residuals of the boundary conditions U0(0) = 0, U0(1) = 1

def solve(tau_guess, C, gamma, M_r, Tr):
  y = np.linspace(0, 1, 1001)
  U0_guess = y                      # Linear guess for U0
  T_guess = np.full_like(y, Tr)     # Constant guess for T
  X_guess = np.vstack((U0_guess, T_guess))    
  
  sol = solve_bvp(
    lambda y, X: ode_system(y, X, tau_guess, C, gamma, M_r),
    lambda Xa, Xb: bc(Xa, Xb, tau_guess, C, gamma, M_r),
    y,
    X_guess,
    max_nodes = len(y) * 100,
    tol = 1e-12,
    verbose = 2
  )
  
  print(sol)
  
  if not sol.success:
    raise Exception(sol.message) #ValueError("Failed to find a solution")
  
  T = sol.y[1]
  T += (1 - T[-1])                      # enforce boundary condition of T0(1) = 1
  eta = T ** (3/2) * (1 + C) / (T + C)  # calculate viscosity
  xi = T * 8.3144 / (28.97)             # calculate specific volume

  return sol.x, sol.y[0], T, eta, xi  # y, U0, T, eta, xi

plt.figure(figsize=(8,7))

styles = ['solid', 'dashed', 'dotted']
# Plot curves for each mach number
for i, M_r in enumerate(M):
  Tr = 1 + (gamma - 1) / 2 * M_r**2  # Recovery temperature  
  y, U0, T, eta, xi = solve(tau_guess, C, gamma, M_r, Tr)    # solve

  # Velocity
  plt.subplot(2, 2, 1)
  plt.axis((0, 1, 0, 1))
  plt.plot(U0, y, label=f'Mr = {M_r}', linestyle=styles[i])
  plt.ylabel('y')
  plt.xlabel('U0')
  plt.title('Velocity')
  plt.legend()

  # Temperature
  plt.subplot(2, 2, 2)
  plt.axis((1, 1.6, 0, 1))
  plt.plot(T, y, label=f'Mr = {M_r}', linestyle=styles[i])
  plt.ylabel('y')
  plt.xlabel('T0')
  plt.title('Temperature')
  plt.legend()
  
  # Specific volume (xi)
  plt.subplot(2, 2, 3)
  plt.axis((1, 1.6, 0, 1))
  plt.plot(T, y, label=f'Mr = {M_r}', linestyle=styles[i])
  plt.ylabel('y')
  plt.xlabel('Xi0')
  plt.title('Specific volume')
  plt.legend()

  # Viscosity coefficient (eta)
  plt.subplot(2, 2, 4)
  plt.axis((0.9, 1.5, 0, 1))
  plt.plot(eta, y, label=f'Mr = {M_r}', linestyle=styles[i])
  plt.ylabel('y')
  plt.xlabel('Eta0')
  plt.title('Viscosity coefficient')
  plt.legend()

plt.tight_layout()
plt.show()