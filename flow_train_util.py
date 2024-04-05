import copy
import functools
import os
import blobfile as bf
from einops import repeat
import torch
import torch.distributed as dist
from tqdm import tqdm
import wandb
from flow_sample_eval_s2s import do_evaluate
from flowseq.rflow import _interp_xt_and_mask
from flowseq.utils import dist_util, logger
from flowseq.utils.nn import update_ema


EPS = 1e-3
SDE_T = 1.0


def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def grad_clip(opt, model, max_grad_norm=2.0):
    if hasattr(opt, "clip_grad_norm"):
        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
        opt.clip_grad_norm(max_grad_norm)
    else:
        # Revert to normal clipping otherwise, handling Apex or full precision
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),  # amp.master_params(self.opt) if self.use_apex else
            max_grad_norm,
        )


class TrainLoop_Flow:
    def __init__(
        self,
        *,
        model,
        flow,
        opt,
        accelerator,
        data_train,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        anneal_lr,
        weight_decay=0.0,
        learning_steps=0,
        checkpoint_path="",
        gradient_clipping=-1.0,
        data_val=None,
        eval_interval=-1,
        args=None,
        **kwargs,
    ):
        self.args = args
        self.model = model
        self.flow = flow
        self.accelerator = accelerator
        self.anneal_lr = anneal_lr
        self.data = data_train
        self.eval_data = data_val
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.weight_decay = weight_decay
        self.learning_steps = learning_steps
        self.gradient_clipping = gradient_clipping
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.checkpoint_path = checkpoint_path
        self.opt = opt
        self.ema_params = [
            copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
        ]
        logger.log("starting training from scratch")

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint and self.accelerator.is_main_process:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                actual_model_path(ema_checkpoint), map_location=dist_util.dev()
            )
            ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if bf.exists(main_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {main_checkpoint}")
            state_dict = dist_util.load_state_dict(
                actual_model_path(main_checkpoint), map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        progress_bar = tqdm(desc="training run_loop", total=self.learning_steps)
        while (
            not self.learning_steps
            or self.step + self.resume_step < self.learning_steps
        ):
            batch_embed, batch_dict = next(self.data)
            self.forward_backward(batch_embed, batch_dict)
            if self.step % self.log_interval == 0 and self.accelerator.is_main_process:
                logger.dumpkvs()
                logger.log(self.checkpoint_path)
            if (
                self.eval_data is not None
                and self.step % self.eval_interval == 0
                and self.step > 0
            ):
                batch_eval, cond_eval = next(self.eval_data)
                self.do_eval(batch_eval, cond_eval)
                if self.accelerator.is_main_process:
                    logger.log(
                        f"eval on validation set, checkpoint_path = {self.checkpoint_path}"
                    )
                    logger.dumpkvs()
            if self.step > 0 and self.step % self.save_interval == 0:
                self.save()
            self.step += 1
            if self.accelerator.is_main_process:
                wandb.log(dict(global_step=self.step))
            progress_bar.update(1)
        if (self.step - 1) % self.save_interval != 0:
            self.save()
        progress_bar.close()

    @torch.no_grad()
    def sample_and_log(
        self,
        batch_dict,
        input_ids_mask,
        model_kwargs,
        ode_stepnum,
        sample_output_path="sample_and_log_dir",
        candicate_num=-1,
    ):
        sample_output_path = os.path.join(
            self.args.checkpoint_path, sample_output_path, f"step{self.step}"
        )

        (
            eval_dict,
            _gen,
            _ref,
            _source,
            samples,
            noise_masked,
        ) = do_evaluate(
            batch_dict=batch_dict,
            input_ids_mask=input_ids_mask,
            model_kwargs=model_kwargs,
            ode_stepnum=ode_stepnum,
            args=self.args,
            _model=self.model,
            _flow=self.flow,
            sample_output_path=sample_output_path,
            candicate_num=candicate_num,
        )

        if True:
            columns = ["source", "reference", "generate"]
            table = wandb.Table(columns=columns)
            for recov, ref, src in zip(_gen, _ref, _source):
                table.add_data(src, ref, recov)
            wandb_dict = {f"gen_text_stepnum{ode_stepnum}": table}
            wandb_dict.update(eval_dict)
            wandb.log(wandb_dict)
        return samples, noise_masked

    @torch.no_grad()
    # @torch.rank_zero_only()
    def cal_straightness(
        self,
        batch_dict,
        sample_N=10,
        sample_T=10,
    ):
        if self.accelerator.is_main_process:
            model_kwargs = {}
            input_ids_mask = batch_dict.pop("input_mask")
            samples, x_noised = self.sample_and_log(
                batch_dict,
                model_kwargs=model_kwargs,
                input_ids_mask=input_ids_mask,
                ode_stepnum=200,
            )

            if True:
                # sampling x from z by a normal ODE.
                # calculate the straightness of the flow by randomly sample several VF from different t.
                # calculate the straightness
                # log the sampled sentences
                samples, x_noised, _input_ids_mask = (
                    samples[:sample_N],
                    x_noised[:sample_N],
                    input_ids_mask[:sample_N],
                )
                assert len(samples) == len(x_noised) and len(x_noised) == sample_N
                samples = repeat(
                    samples, "b seqlen dim -> (b k) seqlen dim", k=sample_T
                )
                x_noised = repeat(
                    x_noised, "b seqlen dim -> (b k) seqlen dim", k=sample_T
                )
                _input_ids_mask = repeat(
                    _input_ids_mask, "b seqlen -> (b k) seqlen", k=sample_T
                )
                straight_t = (
                    torch.rand(sample_N * sample_T, device=x_noised.device)
                    * (SDE_T - EPS)
                    + EPS
                )
                ########################################
                vf_gt = samples - x_noised  # bsz*T, seqlen, dim

                x_t_masked = _interp_xt_and_mask(
                    samples, straight_t, x_0=x_noised, mask=_input_ids_mask
                )

                vf_est = self.model(x_t_masked, straight_t, **model_kwargs)

                straightness = torch.abs(vf_gt - vf_est).mean()
                log_loss_dict(
                    {f"straightness_v2": straightness},
                )
        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def do_eval(self, batch_emb, batch_dict):
        zero_grad(self.model_params)
        for i in range(0, len(batch_emb), self.microbatch):
            micro = batch_emb[i : i + self.microbatch].to(dist_util.dev())
            micro_dict = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in batch_dict.items()
            }
            micro_dict_clone = copy.deepcopy(micro_dict)
            t = (
                torch.rand(len(micro), device=micro.device, dtype=micro.dtype)
                * (SDE_T - EPS)
                + EPS
            )
            compute_losses = functools.partial(
                self.flow.training_losses,
                self.model,
                micro,
                t,
                model_kwargs=micro_dict,
            )

            loss_dict = compute_losses()

            _ = loss_dict.pop("wandb_dict", None)
            loss_dict = {f"eval_{k}": v for k, v in loss_dict.items()}
            log_loss_dict(loss_dict)

        self.cal_straightness(batch_dict=micro_dict_clone)

    def forward_backward(self, batch_embed, batch_dict):
        zero_grad(self.model_params)
        for i in range(0, len(batch_embed), self.microbatch):
            _micro_embed = batch_embed[i : i + self.microbatch].to(dist_util.dev())
            _micro_dict = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in batch_dict.items()
            }
            t = (
                torch.rand(
                    len(_micro_embed),
                    device=_micro_embed.device,
                    dtype=_micro_embed.dtype,
                )
                * (SDE_T - EPS)
                + EPS
            )

            compute_losses = functools.partial(
                self.flow.training_losses,
                self.model,
                _micro_embed,
                t,
                model_kwargs=_micro_dict,
            )

            loss_dict = compute_losses()

            loss = (loss_dict["loss"]).mean()

            wandb_dict = loss_dict.pop("wandb_dict", None)
            wandb_dict.update({k: v for k, v in loss_dict.items()})
            log_loss_dict(wandb_dict)

            loss.backward()
        if self.gradient_clipping > 0:
            grad_clip(self.opt, self.model, max_grad_norm=self.gradient_clipping)
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def _anneal_lr(self):
        if self.anneal_lr == True:
            if not self.learning_steps:
                return
            frac_done = (self.step + self.resume_step) / self.learning_steps
            lr = self.lr * (1 - frac_done)
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr
        else:
            lr = self.opt.param_groups[0]["lr"]
        if self.accelerator.is_main_process:
            logger.logkv_mean("lr", lr)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if self.accelerator.is_main_process:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                print("writing to", bf.join(get_blob_logdir(), filename))
                print("writing to", bf.join(self.checkpoint_path, filename))

                with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f:
                    torch.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    if filename[-3:] == ".pt":
        return int(filename[-9:-3])
    else:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(losses):
    for key, values in losses.items():
        if isinstance(values, torch.Tensor):
            logger.logkv_mean(key, values.mean().item())
        elif isinstance(values, float) or isinstance(values, int):
            logger.logkv_mean(key, values)
        else:
            raise NotImplementedError


def actual_model_path(model_path):
    return model_path
