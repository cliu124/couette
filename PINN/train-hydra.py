from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import rootutils
import tensorflow as tf
from omegaconf import DictConfig

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


def output_fn(outputs: Dict[str, tf.Tensor],
              x: tf.Tensor,
              t: tf.Tensor):
    """Define `output_fn` function that will be applied to outputs of net."""

    return outputs


def pde_fn(outputs: Dict[str, tf.Tensor],
           y: tf.Tensor,
           t: tf.Tensor,
        #    extra_vars: Dict[str, tf.Tensor]
        ):
    """Define the partial differential equations (PDEs)."""
    
    # f1 = (extra_vars["gamma"] - 1) * extra_vars["M_r"] * extra_vars["M_r"]
    
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


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    pinnstf2.utils.extras(cfg)

    # train the model
    metric_dict, _ = pinnstf2.train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=output_fn
    )

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = pinnstf2.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
