from typing import Dict
from time import time

import tensorflow as tf
import numpy as np

import pinnstf2

M_r = 2.
Pr = 0.72  # value for air
gamma = 1.4  # value for air
C = 0.5  # given

f1 = (gamma - 1) * M_r*M_r

T_r = 1 + f1 / 2 * Pr  # Recovery temperature

# PINN settings
epochs = 64000
N_init = 50        # Number of initial samples
N_bound = 100       # Number of boundary samples
N_mesh = 20000    # Number of mesh samples

def read_data_fn(root_path):
  """Read and preprocess data from the specified root path.

  :param root_path: The root directory containing the data.
  :return: Processed data will be used in Mesh class.
  """

  data = pinnstf2.utils.load_data(root_path, "couette_startup_M2.0_T1001_Y501_66a52d93.mat")
#   y = data["y"]
  U = data["U_sol"]
  T = data["T_sol"]
  return {"U": U, "T": T}

# Define spatial and temporal domains
time_domain = pinnstf2.data.TimeDomain(t_interval=[0, 0.5], t_points = 1001)
spatial_domain = pinnstf2.data.Interval(x_interval=[0, 1], shape = [501, 1])

# Construct mesh
mesh = pinnstf2.data.Mesh(root_dir='export',
                          read_data_fn=read_data_fn,
                          spatial_domain = spatial_domain,
                          time_domain = time_domain)


# Sample initial condition
# initial_cond = pinnstf2.data.InitialCondition(mesh = mesh,
#                                       num_sample = N0,
#                                       solution = ['U', 'T'])

# Initial condition
def initial_fun(y):
    U_init = np.zeros_like(y)
    U_init[-1] = 1
    T_init = np.ones_like(y)
    T_init[0] = T_r
    return {'U': U_init, 'T': T_init}
  
initial_cond = pinnstf2.data.InitialCondition(mesh = mesh,
                                            num_sample = N_init,
                                            initial_fun = initial_fun,
                                            solution = ['U', 'T'])

# Periodic boundary condition
# boundary_cond = pinnstf2.data.PeriodicBoundaryCondition(mesh = mesh,
#                                               num_sample = N_b,
#                                               derivative_order = 0,
#                                               solution = ['U', 'T'])

def boundary_fun(t):
    U_lb = np.zeros_like(t)
    U_ub = np.ones_like(t)
    T_lb = np.full_like(t, T_r)
    T_ub = np.ones_like(t)
    U_bound = np.vstack([U_ub, U_lb])
    T_bound = np.vstack([T_ub, T_lb])
    return {'U': U_bound, 'T': T_bound}

boundary_cond = pinnstf2.data.DirichletBoundaryCondition(mesh = mesh,
                                                          num_sample = N_bound,
                                                        #   boundary_fun = boundary_fun,
                                                          solution = ['U', 'T'])

# Collection points and solutions
mesh_sample = pinnstf2.data.MeshSampler(mesh = mesh,
                                   num_sample = N_mesh,
                                   collection_points = ['f_U', 'f_T'])

# validation data
validation_set = pinnstf2.data.MeshSampler(mesh = mesh,
                                    solution = ['U', 'T'])

# define NN
neural_net = pinnstf2.models.FCN(layers = [2, 100, 100, 100, 100, 2],
                          output_names = ['U', 'T'],
                          lb=mesh.lb,
                          ub=mesh.ub)

# output fn
def output_fn(outputs: Dict[str, tf.Tensor],
              x: tf.Tensor,
              t: tf.Tensor):
    """Define `output_fn` function that will be applied to outputs of net."""

    return outputs
  
  
# pde fn
def pde_fn(outputs: Dict[str, tf.Tensor],
           y: tf.Tensor,
           t: tf.Tensor):   
    """Define the partial differential equations (PDEs)."""
    
    U_y, U_t = pinnstf2.utils.gradient(outputs["U"], [y, t])
    T_y, T_t = pinnstf2.utils.gradient(outputs["T"], [y, t])

    # u_xx = pinnstf2.utils.gradient(u_x, x)[0]

    eta = tf.maximum(outputs["T"], 0) ** (3/2) * (1 + C) / (outputs["T"] + C)
    tau = eta * U_y
    tau_y = pinnstf2.utils.gradient(tau, y)
    E_y = pinnstf2.utils.gradient(f1 * tau * outputs["U"] + eta / Pr * T_y, y)

    outputs["f_U"] = U_t - tau_y
    outputs["f_T"] = T_t - E_y

    return outputs

# Define training dataset
train_datasets = [mesh_sample, initial_cond, boundary_cond]
# Define validation dataset
val_dataset = validation_set

datamodule = pinnstf2.data.PINNDataModule(train_datasets = train_datasets,
                                            val_dataset = val_dataset,
                                            pred_dataset = validation_set)

model = pinnstf2.models.PINNModule(net = neural_net,
                                   pde_fn = pde_fn,
                                   output_fn = output_fn,
                                   loss_fn = 'mse',
                                   jit_compile = False)

trainer = pinnstf2.Trainer(max_epochs=epochs, check_val_every_n_epoch=10) # 20000

trainer.fit(model=model, datamodule=datamodule)

trainer.validate(model=model, datamodule=datamodule)

preds_dict = trainer.predict(model=model, datamodule=datamodule)

# plotting
U_exact = mesh.solution["U"]
T_exact = mesh.solution["T"]

U_pred = preds_dict["U"].reshape(U_exact.shape)
T_pred = preds_dict["T"].reshape(T_exact.shape)

# accuracy
U_diff = U_exact - U_pred
U_rmse = np.sqrt(np.mean(U_diff * U_diff))

import os
import logging
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


log = logging.getLogger(__name__)

def figsize(scale, nplots=1):
    """Calculate the figure size based on a given scale and number of plots.

    :param scale: Scaling factor for the figure size.
    :param nplots: Number of subplots in the figure (default is 1).
    :return: Calculated figure size in inches.
    """

    fig_width_pt = 390.0  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = nplots * fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size
def newfig(width, nplots=1):
    """Create a new figure with a specified width and number of subplots.

    :param width: Width of the figure.
    :param nplots: Number of subplots in the figure (default is 1).
    :return: Created figure and subplot axis.
    """

    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax
def savefig(filename, crop=True):
    """Save a figure to the specified filename with optional cropping.

    :param filename: Name of the output file (without extension).
    :param crop: Whether to apply tight cropping to the saved image (default is True).
    """
    dir_name = os.path.dirname(filename)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if crop:
        plt.savefig(f"{filename}.pdf", bbox_inches="tight", pad_inches=0)
        # plt.savefig(f"{filename}.eps", bbox_inches="tight", pad_inches=0)
    else:
        plt.savefig(f"{filename}.pdf")
        # plt.savefig(f"{filename}.eps")

    log.info(f"Image saved at {filename}")

points = [10, 100, 500]

x0, t0, u0 = train_datasets[1][:]
x_b, t_b, _ = train_datasets[2][:]
mid = t_b.shape[0] // 2

X0 = np.hstack((x0[0], t0))
X_ub = np.hstack((x_b[0][:mid], t_b[:mid]))
X_lb = np.hstack((x_b[0][mid:], t_b[mid:]))
X_u_train = np.vstack([X0, X_lb, X_ub])

fig, ax = newfig(2, 1)
ax.axis("off")

# Row 0
# Full soln
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=0.8, bottom=0.5, left=0.2, right=0.8, wspace=0.35)
ax = plt.subplot(gs0[0, 0])#[:, :])

h = ax.imshow(
    U_pred, #U_exact,
    interpolation="nearest",
    cmap="plasma", # YlGnBu
    extent=[mesh.lb[1], mesh.ub[1], mesh.lb[0], mesh.ub[0]],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    "kx",
    label="Data (%d pts)" % (X_u_train.shape[0]),
    markersize=3,
    clip_on=False,
)
line = np.linspace(mesh.spatial_domain_mesh[:].min(), mesh.spatial_domain_mesh[:].max(), 2)[:, None]

ax.plot(mesh.time_domain[points[0]] * np.ones((2, 1)), line, "k--", linewidth=1)
ax.plot(mesh.time_domain[points[1]] * np.ones((2, 1)), line, "k--", linewidth=1)
ax.plot(mesh.time_domain[points[2]] * np.ones((2, 1)), line, "k--", linewidth=1)

ax.set_xlabel("$t$")
ax.set_ylabel("$y$")
leg = ax.legend(frameon=False, loc=(0.6, -0.25)) #"best")
ax.set_title("$U(y, t)$", fontsize=10)

# residuals
ax = plt.subplot(gs0[0, 1])#[:, :])

mpl.colors.Normalize(vmin=-1, vmax=1)
h2 = ax.imshow(
    U_diff,
    interpolation="nearest",
    cmap="seismic", # YlGnBu
    extent=[mesh.lb[1], mesh.ub[1], mesh.lb[0], mesh.ub[0]],
    origin="lower",
    aspect="auto",
    vmin=-1,
    vmax=1
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h2, cax=cax)

ax.set_xlabel("$t$")
ax.set_ylabel("$y$")
ax.set_title(f"Residuals; RMSE={U_rmse:.4f}", fontsize=10)

# Row 1: Slices
# Row 2: Slice residuals

gs1 = gridspec.GridSpec(1, len(points))
gs1.update(top=0.45, bottom=0, left=0.2, right=0.8, wspace=0.35)

gs2 = gridspec.GridSpec(1, len(points))
gs2.update(top=-0.05, bottom=-0.2, left=0.2, right=0.8, wspace=0.35)

for i, point in enumerate(points):
    ax1 = plt.subplot(gs1[0, i])
    ax1.plot(U_exact[:, point], mesh.spatial_domain_mesh[:, point, 0], "b-", linewidth=2, label="Exact")
    ax1.plot(U_pred[:, point], mesh.spatial_domain_mesh[:, point, 0], "r--", linewidth=2, label="Prediction")
    ax1.set_ylabel("$y$")
    ax1.set_xlabel("$U$")
    ax1.set_title("$t = %.2f$" % (mesh.time_domain[point]), fontsize=10)
    ax1.axis("square")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    ax2 = plt.subplot(gs2[0, i])
    ax2.plot(mesh.spatial_domain_mesh[:, point, 0], U_diff[:, point], "g--", linewidth=2, label="Residual")
    # ax2.set_title("Residuals", fontsize=10)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([-1, 1])
    if i == len(points) - 1:
        ax1.legend(loc="upper center", bbox_to_anchor=(0, -0.2), ncol=5, frameon=False)
        ax2.legend(loc="upper center", bbox_to_anchor=(0, -0.25), ncol=5, frameon=False)

name = f"/fig-{epochs}-{round(time())}"
savefig("PINN/out" + name)
print(f"Saved to {name}")

