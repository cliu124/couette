import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
h = 1.0  # Channel height
U = 1.0  # Upper wall velocity
gamma = 1.4  # Specific heat ratio
Pr = 0.72  # Prandtl number
M = 1  # Mach number
C = 0.5  # Sutherland's law constant
T_ref = 1 #273.15  # Reference temperature (K)
Tr = 1 + (gamma - 1) / 2 * M**2  # Recovery temperature

# Numerical parameters
Ny = 101  # Number of spatial grid points
Nt = 10001  # Increased number of time steps
dy = h / (Ny - 1)
t_max = 0.2
dt = t_max / Nt
t = np.linspace(0, t_max, Nt)

# Grid setup
y = np.linspace(0, h, Ny)

# Initialize profiles
u = np.zeros((Nt, Ny))
T = np.ones((Nt, Ny)) * T_ref
T[:, 0] = Tr #T_ref  # Lower wall temperature
T[:, -1] = T_ref  # Upper wall temperature
u[:, -1] = U  # Upper wall velocity

# Helper functions
def safe_power(x, a):
    return np.sign(x) * (np.abs(x) ** a)

def visc(T):
    return safe_power(T/T_ref, 3/2) * (T_ref + C) / (T + C)

def ddx(f):
    return np.gradient(f, dy, edge_order=2)

def d2dx2(f):
    return np.gradient(np.gradient(f, dy, edge_order=2), dy, edge_order=2)

# Main solver
for n in range(1, Nt):
    print(f'\rIteration {n}', end="")
    # Compute viscosity
    mu = visc(T[n-1])
    
    # Solve momentum equation (central difference in space, forward Euler in time)
    u[n, 1:-1] = u[n-1, 1:-1] + dt * (
        ddx(mu * ddx(u[n-1]))[1:-1]
    )
    
    # Solve energy equation (central difference in space, forward Euler in time)
    T[n, 1:-1] = T[n-1, 1:-1] + dt / (gamma * Pr * M**2) * (
        ddx(mu * ddx(T[n-1]))[1:-1] +
        (gamma - 1) * M**2 * mu[1:-1] * (ddx(u[n-1])[1:-1])**2
    )
    
    # Apply boundary conditions
    u[n, 0] = 0
    u[n, -1] = U
    T[n, 0] = Tr #T_ref
    T[n, -1] = T_ref
    
    # Ensure temperature stays positive
    # T[n] = np.maximum(T[n], 0) # 0.1 * T_ref
print("\nDone")
# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Velocity color map
im1 = ax1.imshow(u.T, aspect='auto', origin='lower', extent=[0, t_max, 0, h], # u.T/U
                 cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
ax1.set_title('Velocity Evolution')
ax1.set_xlabel('Time')
ax1.set_ylabel('y/h')
plt.colorbar(im1, ax=ax1, label='u/U')

# Temperature color map
im2 = ax2.imshow(T.T, aspect='auto', origin='lower', extent=[0, t_max, 0, h], # (T.T-T_ref)/T_ref
                 cmap='plasma', interpolation='nearest')
ax2.set_title('Temperature Evolution')
ax2.set_xlabel('Time')
ax2.set_ylabel('y/h')
plt.colorbar(im2, ax=ax2, label='(T-T_ref)/T_ref')

plt.tight_layout()
plt.show()

# Print final profiles
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(u[-1], y/h) # /U
plt.title('Final Velocity Profile')
plt.xlabel('u/U')
plt.ylabel('y/h')

plt.subplot(122)
plt.plot(T[-1], y/h)# (T[-1]-T_ref)/T_ref
plt.title('Final Temperature Profile')
plt.xlabel('(T-T_ref)/T_ref')
plt.ylabel('y/h')

plt.tight_layout()
plt.show()