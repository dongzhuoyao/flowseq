"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
from einops import rearrange, repeat

import numpy as np
from scipy import integrate
import torch as th
import torch
import sys
import torch.distributed as dist

from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
import wandb
from flowseq.sampling_rflow import (
    from_flattened_numpy,
    to_flattened_numpy,
)

sys.path.append(".")

import torch.nn.functional as F

from .utils.nn import mean_flat


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred, mask=None):
    _condition_num = (mask == 0).sum().item()

    correct = (
        torch.eq(y_true, y_pred).sum().item()
    )  # torch.eq() calculates where two tensors are equal
    acc = ((correct - _condition_num) / (len(y_pred) - _condition_num)) * 100
    return acc


@torch.no_grad()
def euler_sampler(model, z=None):
    raise NotImplementedError("Not carefully checked yet")
    with torch.no_grad():
        # Initial sample
        if z is None:
            z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
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


@torch.no_grad()
def rflow_sampler(
    model,
    sde,
    clip_denoised,
    denoised_fn,
    x_0=None,
    model_kwargs=None,
    top_p=None,
    clamp_step=None,
    clamp_first=None,
    mask=None,
    x_embed=None,
    ode_package="scipy",
    ode_stepnum=None,
    **kwargs,
):
    assert ode_package in ["torchdiffeq", "scipy"]
    device, shape = x_0.device, x_0.shape
    rtol = atol = sde.ode_tol
    eps = sde.eps

    def mask_and_clip_per_step(
        x_t,
        mask,
        x_1,
        t,
    ):
        assert mask is not None

        x_t = th.where(mask == 0, x_1, x_t)
        if clip_denoised:
            raise NotImplementedError("not use it any more")
            x_1_est = (x_t - x_0) / (1 - t + 1e-10) + x_0
            x_1_rounded = denoised_fn(x_1_est, t)
            x_t_rouned = x_1_rounded * t + x_0 * (1 - t)
            return x_t_rouned
        else:
            return x_t

    if ode_package == "torchdiffeq":
        progress_bar = tqdm(desc="sampling torchdiffeq")

        def ode_func(t, x_t):
            # print("torchdiffeq ode time", t)
            t_tensor = torch.ones(len(x_0), device=x_t.device) * t
            #########
            x_t = mask_and_clip_per_step(x_t, mask, x_1=x_embed, t=t)
            #########
            vf_t = model(t_tensor, x_t)

            if False:
                input_ids_x = model_kwargs["input_ids"].to(device)
                input_ids_mask = model_kwargs["input_mask"].to(device)
                get_logits_fn = model_kwargs["get_logits"]

                with open("backward_acc.txt", "a") as f:
                    x_1_est = (
                        x_t + (1 - repeat(t.reshape(-1), "1->b 1 1", b=len(x_0))) * vf_t
                    )
                    logits = get_logits_fn(x_1_est)  # bsz, seqlen, vocab
                    acc = accuracy_fn(
                        y_pred=logits.view(-1, logits.size(-1)).max(1).indices,
                        y_true=input_ids_x.view(-1),
                        mask=input_ids_mask,
                    )
                    f.write(f"{t},{acc}\n")
                    f.flush()

            progress_bar.update(int(t * 1000))
            return vf_t

        result = odeint(
            ode_func,
            x_0,
            torch.tensor([0, sde.T], device=x_0.device, dtype=x_0.dtype),
            method="euler",
            rtol=rtol,
            atol=atol,
            adjoint_params=(),
            options=dict(step_size=sde.T / ode_stepnum),
        )[-1]
        progress_bar.close()
        return result, ode_stepnum

    elif ode_package == "scipy":
        raise NotImplementedError(
            "not use it any more, need to support specific step size"
        )
        method = "RK45"

        def ode_func(t, x_t):
            x_t = from_flattened_numpy(x_t, shape).to(device).type(torch.float32)
            t_tensor = torch.ones(len(x_0), device=x_t.device) * t

            #########
            x_t = mask_and_clip_per_step(x_t, mask, x_1=x_embed, t=t)
            #########

            vf_t = model(t_tensor, x_t)
            vf_t_flat = to_flattened_numpy(vf_t)
            # print(f"{ode_package} ode time", t)

            return vf_t_flat

        solution = integrate.solve_ivp(
            ode_func,
            (eps, sde.T),
            to_flattened_numpy(x_0),
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

        return x, nfe
    elif ode_package == "euler_raw":
        steps = 1 // ode_stepnum
        x_t = None
        for i in np.linspace(0, 1, steps):
            # x_0 = x_0 + model(x_0, i * sde.T / steps) * sde.T / steps
            t = i * sde.T / steps
            t_tensor = torch.ones(len(x_0), device=x_t.device) * t
            #########
            x_t = mask_and_clip_per_step(x_t, mask, x_1=x_embed, t=t)
            #########
            vf_t = model(t_tensor, x_t)

    else:
        raise NotImplementedError(f"ode_package {ode_package} not implemented")


def _interp_xt_and_mask(x_1, t, x_0, mask=None):
    assert x_0.shape == x_1.shape
    t_expand = t[:, None, None]

    x_t = t_expand * x_1 + (1.0 - t_expand) * x_0

    if mask == None:
        return x_t
    else:
        mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_1.shape)
        return th.where(mask == 0, x_1, x_t)


class RFlow:
    def __init__(
        self,
        *kwargs,
    ):
        pass

    def _get_x_start(self, x_start_mean, std):
        """
        Word embedding projection from {Emb(w)} to {x_0}
        :param x_start_mean: word embedding
        :return: x_0
        """
        noise = th.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        # print(x_start_mean.device, noise.device)
        return x_start_mean + std * noise

    def training_losses(self, model, *args, **kwargs):
        self.model = model
        return self.training_losses_seq2seq(model, *args, **kwargs)

    def _token_discrete_loss(
        self,
        x_t,
        get_logits,
        input_ids,
        mask=None,
    ):
        """
        the loss of -log p(w|z_0)
        :param x_start_mean: word embedding
        :return: x_0
        """

        assert mask is not None  # in this conditional generation task
        logits = get_logits(x_t)  # bsz, seqlen, vocab
        # print(logits.shape)
        loss_fct = th.nn.CrossEntropyLoss(reduction="none")
        decoder_nll = loss_fct(
            logits.view(-1, logits.size(-1)), input_ids.view(-1)
        ).view(input_ids.shape)

        acc = accuracy_fn(
            y_pred=logits.view(-1, logits.size(-1)).max(1).indices,
            y_true=input_ids.view(-1),
            mask=mask,
        )

        if mask != None:
            decoder_nll *= mask
            # print(decoder_nll.shape)
            decoder_nll = decoder_nll.sum(dim=-1) / mask.sum(dim=-1)
        else:
            decoder_nll = decoder_nll.mean(dim=-1)

        return decoder_nll, acc

    def decode(
        self,
        model,
        noise,
        **kwargs,
    ):
        class _SDE:
            def __init__(
                self,
            ):
                self.ode_tol = 1e-5
                self.T = 1
                self.eps = 1e-3

        func = lambda t, x: model(x, t)

        result, _nfe = rflow_sampler(model=func, x_0=noise, sde=_SDE(), **kwargs)
        return result, _nfe

    def cal_triangular_reg(
        self, model, x_1, x_0, model_kwargs, input_ids_x, input_ids_mask, get_logits_fn
    ):
        _bs = len(x_1)
        bs_1_3 = _bs // 3
        _x_1, _x_0, _input_ids_mask, _input_ids_x = (
            x_1[:bs_1_3],
            x_0[:bs_1_3],
            input_ids_mask[:bs_1_3],
            input_ids_x[:bs_1_3],
        )
        _t_mix = th.rand(bs_1_3 * 3, device=_x_1.device)
        _x_0 = _x_0.repeat(3, 1, 1)
        _x_1 = _x_1.repeat(3, 1, 1)

        vf_gt = _x_1 - _x_0

        x_t_concat = _interp_xt_and_mask(
            _x_1, _t_mix, x_0=_x_0, mask=_input_ids_mask.repeat(3, 1)
        )
        t_concat = _t_mix

        vf_pred = model(x_t_concat, t_concat, **model_kwargs)

        x_1_est = x_t_concat + (1 - rearrange(_t_mix, "b->b 1 1")) * vf_pred
        x_1_est = x_1_est[:bs_1_3]
        terms = {}
        terms["mse_triangle"] = mean_flat((vf_pred - vf_gt) ** 2)

        terms["nll_masked"], terms["x_1_est_acc"] = self._token_discrete_loss(
            x_1_est,
            get_logits_fn,
            _input_ids_x,
            mask=_input_ids_mask,
        )

        terms["loss"] = terms["mse_triangle"].mean() + terms["nll_masked"].mean()

        wandb_dict = self.log_min_max(x_1, x_1_est)
        terms["wandb_dict"] = wandb_dict

        return terms

    def training_losses_seq2seq(self, model, x_1, t, model_kwargs=None, x_0=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs. # not used unless fixing the input embeddings
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        assert "input_ids" in model_kwargs
        input_ids_x = model_kwargs.pop("input_ids").to(t.device)
        input_ids_mask = model_kwargs.pop("input_mask").to(t.device)
        x_1_mean = model.module.get_embeds(input_ids_x)
        get_logits_fn = model.module.get_logits

        if True:

            x_1 = x_1_mean

            if x_0 is None:
                x_0 = th.randn_like(x_1)

            x_t_masked = _interp_xt_and_mask(
                x_1, t, x_0=x_0, mask=input_ids_mask
            )  # t * x_1 + (1.0 - t) * x_0

            terms = {}

            vf_gt = x_1 - x_0
            vf_pred = model(x_t_masked, t, **model_kwargs)

            terms["mse"] = mean_flat((vf_gt - vf_pred) ** 2)
            x_1_est = x_t_masked + (1 - rearrange(t, "b->b 1 1")) * vf_pred

            terms["nll_masked"], terms["x_1_est_acc_pure"] = self._token_discrete_loss(
                x_1_est,
                get_logits_fn,
                input_ids_x,
                mask=input_ids_mask,
            )
            terms["nll_masked"] = terms["nll_masked"] * 1.0  # nll_weight
            terms["loss"] = terms["mse"].mean() + terms["nll_masked"].mean()
            terms["wandb_dict"] = self.log_min_max(x_1, x_1_est)
            return terms

    def log_min_max(self, x_1, x_1_est):
        x_1_np = x_1.detach().cpu().numpy()
        x_1_est_np = x_1_est.detach().cpu().numpy()

        _wandb_dict = dict(
            x_1_min=x_1_np.min().item(),
            x_1_max=x_1_np.max().item(),
            x_1_mean=x_1_np.mean().item(),
            x_1_std=x_1_np.std().item(),
            x_1_est_min=x_1_est_np.min().item(),
            x_1_est_max=x_1_est_np.max().item(),
            x_1_est_mean=x_1_est_np.mean().item(),
            x_1_est_std=x_1_est_np.std().item(),
        )
        return _wandb_dict
