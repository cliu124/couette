from typing import Dict

import tensorflow as tf
import numpy as np

import pinnstf2

M_r = 2.
Pr = 0.72  # value for air
gamma = 1.4  # value for air
C = 0.5  # given

f1 = (gamma - 1) * M_r*M_r

T_r = 1 + f1 / 2 * Pr  # Recovery temperature

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

N0 = 50

# Sample initial condition
initial_cond = pinnstf2.data.InitialCondition(mesh = mesh,
                                      num_sample = N0,
                                      solution = ['U', 'T'])

# # Initial condition
# def initial_fun(y):
#     U_init = np.zeros_like(y)
#     U_init[-1] = 1
#     T_init = np.ones_like(y)
#     T_init[-1] = T_r
#     return {'U': U_init, 'T': T_init}
  
# in_c = pinnstf2.data.InitialCondition(mesh = mesh,
#                                       num_sample = N0,
#                                       initial_fun = initial_fun,
#                                       solution = ['U', 'T'])

# Periodic boundary condition
N_b = 50
periodic_bound = pinnstf2.data.PeriodicBoundaryCondition(mesh = mesh,
                                              num_sample = 50,
                                              derivative_order = 0,
                                              solution = ['U', 'T'])

# Collection points and solutions
N_f = 20000
mesh_sample = pinnstf2.data.MeshSampler(mesh = mesh,
                                   num_sample = N_f,
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
train_datasets = [mesh_sample, initial_cond, periodic_bound]
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

trainer = pinnstf2.Trainer(max_epochs=20000, check_val_every_n_epoch=1000) # 20000

trainer.fit(model=model, datamodule=datamodule)

trainer.validate(model=model, datamodule=datamodule)

preds_dict = trainer.predict(model=model, datamodule=datamodule)

# plotting
U_exact = mesh.solution["U"]
T_exact = mesh.solution["T"]

U_pred = preds_dict["U"].reshape(U_exact.shape)
T_pred = preds_dict["T"].reshape(T_exact.shape)

# pinnstf2.utils.plot_schrodinger(mesh, U_pred, train_datasets, val_dataset, "couette-out")
import os
import logging
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
        plt.savefig(f"{filename}.eps", bbox_inches="tight", pad_inches=0)
    else:
        plt.savefig(f"{filename}.pdf")
        plt.savefig(f"{filename}.eps")

    log.info(f"Image saved at {filename}")

points = [100, 500, 900]


# Row 1: u(t,x) slices
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

x0, t0, u0 = train_datasets[1][:]
x_b, t_b, _ = train_datasets[2][:]
mid = t_b.shape[0] // 2

X0 = np.hstack((x0[0], t0))
X_ub = np.hstack((x_b[0][:mid], t_b[:mid]))
X_lb = np.hstack((x_b[0][mid:], t_b[mid:]))
X_u_train = np.vstack([X0, X_lb, X_ub])

fig, ax = newfig(1.0, 0.9)
ax.axis("off")

# Row 0: full
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(
    U_pred, #U_exact,
    interpolation="nearest",
    cmap="YlGnBu",
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
    label="Data (%d points)" % (X_u_train.shape[0]),
    markersize=2,
    clip_on=False,
)
line = np.linspace(mesh.spatial_domain_mesh[:].min(), mesh.spatial_domain_mesh[:].max(), 2)[:, None]

ax.plot(mesh.time_domain[points[0]] * np.ones((2, 1)), line, "k--", linewidth=1)
ax.plot(mesh.time_domain[points[1]] * np.ones((2, 1)), line, "k--", linewidth=1)
ax.plot(mesh.time_domain[points[2]] * np.ones((2, 1)), line, "k--", linewidth=1)

ax.set_xlabel("$t$")
ax.set_ylabel("$y$")
leg = ax.legend(frameon=False, loc=(0.6, -0.5)) #"best")
ax.set_title("$U(y, t)$", fontsize=10)

# Row 1: slices
# U_exact = np.flip(U_exact, 0)

gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1 - 1 / 2, bottom=0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(U_exact[:, points[0]], mesh.spatial_domain_mesh[:, points[0], 0], "b-", linewidth=2, label="Exact")
ax.plot(U_pred[:, points[0]], mesh.spatial_domain_mesh[:, points[0], 0], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$y$")
ax.set_xlabel("$U(y, t)$")
ax.set_title("$t = %.2f$" % (mesh.time_domain[points[0]]), fontsize=10)
ax.axis("square")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

ax = plt.subplot(gs1[0, 1])
ax.plot(U_exact[:, points[1]], mesh.spatial_domain_mesh[:, points[1], 0], "b-", linewidth=2, label="Exact")
ax.plot(U_pred[:, points[1]], mesh.spatial_domain_mesh[:, points[1], 0], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$y$")
ax.set_xlabel("$U(y, t)$")
ax.axis("square")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_title("$t = %.2f$" % (mesh.time_domain[points[1]]), fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(U_exact[:, points[2]], mesh.spatial_domain_mesh[:, points[2], 0], "b-", linewidth=2, label="Exact")
ax.plot(U_pred[:, points[2]], mesh.spatial_domain_mesh[:, points[2], 0], "r--", linewidth=2, label="Prediction")
ax.set_ylabel("$y$")
ax.set_xlabel("$U(y, t)$")
ax.axis("square")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_title("$t = %.2f$" % (mesh.time_domain[points[2]]), fontsize=10)

savefig("couette-out" + "/fig")

