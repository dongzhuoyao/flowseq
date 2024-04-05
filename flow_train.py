from datetime import datetime
import os
from flowseq.utils import logger
from flowseq.text_datasets import load_data_text

from basic_utils import (
    create_model_and_flow,
    args_to_dict,
    load_model_emb,
    load_tokenizer,
)
from flow_train_util import TrainLoop_Flow
from transformers import set_seed
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import accelerate
from accelerate.utils import AutocastKwargs
from torch.optim import AdamW


def update_cfg(cfg):
    if cfg.is_debug:
        logger.log("### Debug mode is on")
        cfg.batch_size = 128
        cfg.microbatch = 64
        cfg.learning_steps = 102
        cfg.save_interval = 20
        cfg.log_interval = 10
    else:
        logger.log("### Debug mode is off")

    cfg.checkpoint_path = cfg.checkpoint_path + "_" + cfg.note
    cfg.checkpoint_path = os.path.join(HydraConfig.get().run.dir, cfg.checkpoint_path)

    if os.path.exists(cfg.checkpoint_path):
        cfg.checkpoint_path = cfg.checkpoint_path + f"_{datetime.now():%Y%m%d-%H:%M:%S}"
        print("checkpoint path already exists,renaming")
    print(f"checkpoint path: {cfg.checkpoint_path}")
    return cfg


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg):
    cfg = cfg.data  # hydra
    cfg = update_cfg(cfg)
    set_seed(cfg.seed)
    logger.configure(format_strs=["log", "stdout", "csv"])
    logger.log("### Creating data loader...")
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    kwargs = AutocastKwargs(enabled=False)
    # https://github.com/pytorch/pytorch/issues/40497#issuecomment-709846922
    # https://github.com/huggingface/accelerate/issues/2487#issuecomment-1969997224
    accelerator = accelerate.Accelerator(
        kwargs_handlers=[kwargs],
        mixed_precision=None,
    )
    device = accelerator.device
    accelerate.utils.set_seed(666, device_specific=True)
    rank = accelerator.state.process_index
    print(
        f"Starting rank={rank}, world_size={accelerator.state.num_processes}, device={device}."
    )
    is_multiprocess = True if accelerator.state.num_processes > 1 else False

    tokenizer = load_tokenizer(cfg)
    model_embed = load_model_emb(cfg, vocab_size=tokenizer.vocab_size)

    data_train = load_data_text(
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        data_args=cfg,
        split="train",
        loaded_vocab=tokenizer,
        model_emb=model_embed,  # use model's weights as init
        is_debug=cfg.is_debug,
    )

    data_val = load_data_text(
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        data_args=cfg,
        split="valid",
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=model_embed,  # using the same embedding wight with tranining data
    )

    logger.log(f"### Creating model and flow..., size of vocab {cfg.vocab_size}")

    model, _flow = create_model_and_flow(**args_to_dict(cfg, cfg.keys()))

    next(data_train)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DiffuSeq"),
            name=cfg.checkpoint_path,
            reinit=True,
            job_type="train",
            mode="online",
        )
        wandb.config.update(args_to_dict(cfg, cfg.keys()))

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    data_train, data_val, opt, model = accelerator.prepare(
        data_train, data_val, opt, model
    )

    TrainLoop_Flow(
        model=model,
        flow=_flow,
        opt=opt,
        accelerator=accelerator,
        data_train=data_train,
        data_val=data_val,
        args=cfg,
        **cfg,
    ).run_loop()


if __name__ == "__main__":
    main()
