import copy
import functools
import os, json


import torch as th
import torch
import hydra

from transformers import set_seed
from flowseq.rounding import denoised_fn_round
from flowseq.text_datasets import load_data_text
from pathlib import Path
from tqdm import tqdm

import time
from flowseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    args_to_dict,
    create_model_and_flow,
    load_tokenizer,
)
from eval_utils import _evaluate


def do_evaluate(
    batch_dict,
    input_ids_mask,
    model_kwargs,
    ode_stepnum,
    args,
    _model,
    _flow,
    sample_output_path="sample_and_log_dir",
    candicate_num=10,
):
    if args.is_debug:
        candicate_num = 2
        print("### Debug mode is on, candicate_num", candicate_num)

    os.makedirs(sample_output_path, exist_ok=True)

    input_ids_x = batch_dict.pop("input_ids").to(dist_util.dev())
    x_embed = _model.get_embeds(input_ids_x)
    input_ids_mask_ori = input_ids_mask
    input_ids_mask = th.broadcast_to(
        input_ids_mask.unsqueeze(dim=-1), x_embed.shape
    ).to(dist_util.dev())
    tokenizer = load_tokenizer(args)
    model_emb = (
        th.nn.Embedding(
            num_embeddings=tokenizer.vocab_size,
            embedding_dim=args.hidden_dim,
            _weight=_model.word_embedding.weight.clone().cpu(),
        )
        .eval()
        .requires_grad_(False)
    )

    for _C in range(candicate_num):
        _start_time = time.time()
        noise = th.randn_like(x_embed)
        noise_masked = th.where(input_ids_mask == 0, x_embed, noise)
        logger.log("begin sampling", noise_masked.shape)
        samples, _nfe = _flow.decode(
            _model,
            noise=noise_masked,
            denoised_fn=functools.partial(denoised_fn_round, args, model_emb),
            model_kwargs=model_kwargs,
            mask=input_ids_mask,
            x_embed=x_embed,
            ode_package="torchdiffeq",
            ode_stepnum=ode_stepnum,
            clip_denoised=args.eval.clip_denoised,
        )
        logger.log("end sampling, spend time", time.time() - _start_time)

        logits = _model.get_logits(samples)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []

        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            len_x = args.seq_len - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            len_x = args.seq_len - sum(input_mask).tolist()
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        if True:
            for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
                len_x = args.seq_len - sum(input_mask).tolist()
                tokens = tokenizer.decode_token(seq[len_x:])
                word_lst_recover.append(tokens)

            for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
                len_x = args.seq_len - sum(input_mask).tolist()
                word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
                word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

            json_file = os.path.join(sample_output_path, f"sample_{_C}.json")
            with open(json_file, "a") as fout:
                for recov, ref, src in zip(
                    word_lst_recover, word_lst_ref, word_lst_source
                ):
                    print(
                        json.dumps({"recover": recov, "reference": ref, "source": src}),
                        file=fout,
                    )

            print(f"### Written the decoded output to {json_file}")

    if True:
        eval_dict = _evaluate(
            _folder=sample_output_path,
            _mbr=True,
            _eos="[SEP]",
            _sos="[CLS]",
            _sep="[SEP]",
            _pad="[PAD]",
        )
        eval_dict = {f"metric_{k}": v for k, v in eval_dict.items()}

        return (
            eval_dict,
            word_lst_recover,
            word_lst_ref,
            word_lst_source,
            samples,
            noise_masked,
        )


def update_configs(config):
    config.checkpoint_path = config.checkpoint_path + config.note
    assert config.eval.candidate_num > 0
    return config


@torch.no_grad()
@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg):
    cfg = cfg.data  # hydra
    cfg = update_configs(cfg)

    output_path = os.path.join(
        Path(cfg.eval.model_path).parent,
        f"debug{int(cfg.eval.is_debug)}_eulerstepsize{cfg.eval.ode_stepnum}_can{cfg.eval.candidate_num}_anchor{int(cfg.eval.clip_denoised)}_samples",
    )
    if os.path.exists(output_path):
        output_path += time.strftime("%Y%m%d-%H:%M:%S")

    os.makedirs(output_path, exist_ok=True)

    tokenizer = load_tokenizer(cfg)

    dist_util.setup_dist()
    logger.configure()

    logger.log("### Creating model and flow...")
    logger.log(cfg)
    logger.log("load model from", cfg.eval.model_path)

    _model, _flow = create_model_and_flow(**args_to_dict(cfg, cfg.keys()))
    _model.load_state_dict(
        dist_util.load_state_dict(cfg.eval.model_path, map_location="cpu")
    )
    _model.eval().requires_grad_(False).to(dist_util.dev())

    model_emb = (
        th.nn.Embedding(
            num_embeddings=tokenizer.vocab_size,
            embedding_dim=cfg.hidden_dim,
            _weight=_model.word_embedding.weight.clone().cpu(),
        )
        .eval()
        .requires_grad_(False)
    )
    model_emb.to(dist_util.dev())

    data4eval = load_data_text(
        batch_size=cfg.eval.batch_size,
        seq_len=cfg.seq_len,
        deterministic=True,
        data_args=cfg,
        split=cfg.eval.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(),  # using the same embedding wight with training data
        loop=False,
    )
    batch, _ = next(data4eval)  # first element is empty, don't know why
    print(batch.shape)

    all_test_data = []
    try:
        while True:
            _, batch_dict = next(data4eval)
            all_test_data.append(batch_dict)
    except StopIteration:
        print("### End of reading iteration...")

    print(
        "### Total number of batches",
        sum([_d["input_ids"].shape[0] for _d in all_test_data]),
    )

    for _ith in range(cfg.eval.candidate_num):
        _current_seed = cfg.eval.seed + _ith
        set_seed(_current_seed)

        sample_output_path = os.path.join(
            output_path,
            f"seed{_current_seed}_clampstep{cfg.eval.clamp_step}.json",
        )
        word_lst_recover, word_lst_ref, word_lst_source = [], [], []

        for _i, _batch_dict in enumerate(all_test_data):
            batch_dict = copy.deepcopy(_batch_dict)
            if not batch_dict:
                continue

            if cfg.eval.is_debug and _i >= 1:
                logger.log("### Debug mode is on, break..")
                break

            input_ids_x = batch_dict.pop("input_ids").to(dist_util.dev())
            x_embed = _model.get_embeds(input_ids_x)
            input_ids_mask_ori = input_ids_mask = batch_dict.pop("input_mask")

            noise = th.randn_like(x_embed)
            input_ids_mask = th.broadcast_to(
                input_ids_mask.unsqueeze(dim=-1), x_embed.shape
            ).to(dist_util.dev())
            noise_masked = th.where(input_ids_mask == 0, x_embed, noise)

            model_kwargs = copy.deepcopy(_batch_dict)
            model_kwargs["get_logits"] = _model.get_logits

            samples, _nfe = _flow.decode(
                _model,
                noise=noise_masked,
                clip_denoised=cfg.eval.clip_denoised,
                denoised_fn=partial(denoised_fn_round, cfg, model_emb),
                model_kwargs=model_kwargs,
                top_p=cfg.eval.top_p,
                clamp_step=cfg.eval.clamp_step,
                clamp_first=True,
                ode_package="torchdiffeq",
                ode_stepnum=cfg.eval.ode_stepnum,
                mask=input_ids_mask,
                x_embed=x_embed,
            )
            print("nfe", _nfe)

            logits = _model.get_logits(samples)  # bsz, seqlen, vocab
            cands = th.topk(logits, k=1, dim=-1)

            for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
                len_x = cfg.seq_len - sum(input_mask).tolist()
                tokens = tokenizer.decode_token(seq[len_x:])
                word_lst_recover.append(tokens)

            for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
                len_x = cfg.seq_len - sum(input_mask).tolist()
                word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
                word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        with open(sample_output_path, "a") as fout:
            for recov, ref, src in zip(word_lst_recover, word_lst_ref, word_lst_source):
                print(
                    json.dumps(
                        {
                            "recover": recov,
                            "reference": ref,
                            "source": src,
                            "nfe": _nfe,
                        }
                    ),
                    file=fout,
                )

        print(f"### Written the decoded output to {sample_output_path}")

    if True:
        _evaluate(
            _folder=output_path,
            _mbr=True,
            _eos="[SEP]",
            _sos="[CLS]",
            _sep="[SEP]",
            _pad="[PAD]",
        )


if __name__ == "__main__":
    main()
