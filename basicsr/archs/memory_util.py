import math
import numpy as np
import torch
from typing import Optional


def get_similarity(mk, ms, qk, qe):
    # used for training/inference and memory reading/memory potentiation
    """
    current_key : b, c, h, w
    current_selection: b, c, h, w
    memory_key: b, thw, c
    memory_shrinkage: b, thw, 1
    memory_value: b, c, thw
    """
    # Dimensions in [] are flattened
    CK = mk.shape[2]
    # mk = mk.flatten(start_dim=2) # B x CK x (T*H*W)
    # ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None # B x (T*H*W) x 1
    qk = qk.flatten(start_dim=2) # B x CK x (HW/P)
    qe = qe.flatten(start_dim=2) if qe is not None else None # B x CK x (HW/P)

    if qe is not None:
    #     # See appendix for derivation
    #     # or you can just trust me ヽ(ー_ー )ノ
        # mk = mk.transpose(1, 2) # B x (T*H*W) x CK
        a_sq = (mk.pow(2) @ qe) # B x (T*H*W) x (HW/P)
        two_ab = 2 * (mk @ (qk * qe)) # B x (T*H*W) x (HW/P)
        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True) # B x 1 x (HW/P)
        similarity = (-a_sq+two_ab-b_sq) # B x (T*H*W) x (HW/P)
    else:
        # similar to STCN if we don't have the selection term
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        two_ab = 2 * (mk.transpose(1, 2) @ qk)
        similarity = (-a_sq+two_ab)
        # a_sq = (mk.pow(2) @ qe)

    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)   # B*N*(HW/P)
    else:
        similarity = similarity / math.sqrt(CK)   # B*N*(HW/P)

    return similarity

def do_softmax(similarity, top_k: Optional[int]=None, inplace=False, return_usage=False):
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1) # B*top_k*(HW/P)

        x_exp = values.exp_() # B*top_k*(HW/P)
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True) # B*top_k*(HW/P)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp) # B*N*(HW/P)
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp) # B*N*(HW/P)
    else:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0] # B*1*HW
        x_exp = torch.exp(similarity - maxes) # B*N*HW
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True) # B*1*HW
        affinity = x_exp / x_exp_sum # B*N*HW
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2) # affinity: B*N*(HW/P), usage: B*N

    return affinity

def get_affinity(mk, ms, qk, qe):
    """
    query_key       : B * CK * H * W
    query_selection : B * CK * H * W
    memory_key      : B * CK * T * H * W
    memory_shrinkage: B * 1  * T * H * W
    """
    # shorthand used in training with no top-k
    similarity = get_similarity(mk, ms, qk, qe) # B*(TxHW)*HW
    affinity = do_softmax(similarity) # B*(TxHW)*HW
    return affinity

def readout(affinity, mv):
    """
    affinity: B * (T*H*W) * (HW)
    mv: B * C * THW
    """
    # B, CV, T, H, W = mv.shape

    # mo = mv.view(B, CV, T*H*W)
    mem = torch.bmm(mv, affinity) # B*CV*HW
    # mem = mem.view(B, CV, H, W) # B*(num_objects * CV)*H*W

    return mem
