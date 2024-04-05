# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc


from scipy import integrate


from tqdm import tqdm


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
      model: The score model.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(x, labels):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.

        Returns:
          A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_rectified_flow_sampler(
    sde,
    noise,
    inverse_scaler,
    device="cuda",
    clip_denoised=None,
    denoised_fn=None,
    model_kwargs=None,
    top_p=None,
    clamp_step=None,
    clamp_first=None,
    mask=None,
    x_start=None,
    **kwargs
):
    """
    Get rectified flow sampler

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    shape = noise.shape

    def euler_sampler(model, z=None):
        """The probability flow ODE sampler with simple Euler discretization.

        Args:
          model: A velocity model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        raise NotImplementedError("Not carefully checked yet")
        with torch.no_grad():
            # Initial sample
            if z is None:
                z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(
                    device
                )
                x = z0.detach().clone()
            else:
                x = z

            model_fn = get_model_fn(model, train=False)

            ### Uniform
            dt = 1.0 / sde.sample_N
            eps = 1e-3  # default: 1e-3
            for i in range(sde.sample_N):
                num_t = i / sde.sample_N * (sde.T - eps) + eps
                t = torch.ones(shape[0], device=device) * num_t
                pred = model_fn(x, t)  ### Copy from models/utils.py

                # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability
                sigma_t = sde.sigma_t(num_t)
                pred_sigma = pred + (sigma_t**2) / (
                    2 * (sde.noise_scale**2) * ((1.0 - num_t) ** 2)
                ) * (
                    0.5 * num_t * (1.0 - num_t) * pred
                    - 0.5 * (2.0 - num_t) * x.detach().clone()
                )

                x = (
                    x.detach().clone()
                    + pred_sigma * dt
                    + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)
                )

            x = inverse_scaler(x)
            nfe = sde.sample_N
            return x, nfe

    def rk45_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
          model: A velocity model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            rtol = atol = sde.ode_tol
            method = "RK45"
            eps = 1e-3

            # Initial sample
            if z is None:
                raise NotImplementedError("Not carefully checked yet")
                z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(
                    device
                )
                x = z0.detach().clone()
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = model(vec_t, x)

                _x_t = to_flattened_numpy(drift)

                return _x_t

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func,
                (eps, sde.T),
                to_flattened_numpy(x),
                rtol=rtol,
                atol=atol,
                method=method,
            )
            nfe = solution.nfev
            x = (
                torch.tensor(solution.y[:, -1])
                .reshape(shape)
                .to(device)
                .type(torch.float32)
            )

            x = inverse_scaler(x)

            return x, nfe

    return rk45_sampler

    # print("Type of Sampler:", sde.use_ode_sampler)
    # if sde.use_ode_sampler == "rk45":
    #    return rk45_sampler
    # elif sde.use_ode_sampler == "euler":
    #    return euler_sampler
    # else:
    #    assert False, "Not Implemented!"
