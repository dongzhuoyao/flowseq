import torch
import numpy as np




def get_efficient_knn(model_emb, text_emb):
    emb_norm = (model_emb**2).sum(-1).view(-1, 1)  # vocab
    text_emb_t = torch.transpose(
        text_emb.view(-1, text_emb.size(-1)), 0, 1
    )  # d, bsz*seqlen
    arr_norm = (text_emb**2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
    # print(emb_norm.shape, arr_norm.shape)
    dist = (
        emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t)
    )  # (vocab, d) x (d, bsz*seqlen)
    dist = torch.clamp(dist, 0.0, np.inf)
    # print(dist.shape)
    topk_out = torch.topk(-dist, k=1, dim=0)
    return topk_out.values, topk_out.indices


def denoised_fn_round(args, model, text_emb, t):
    # print(text_emb.shape) # bsz, seqlen, dim
    model_emb = model.weight  # input_embs
    _shape, _device = text_emb.shape, text_emb.device

    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb

    val, indices = get_efficient_knn(model_emb, text_emb.to(model_emb.device))
    rounded_tokens = indices[0]

    new_embeds = model(rounded_tokens).view(_shape).to(_device)

    return new_embeds
