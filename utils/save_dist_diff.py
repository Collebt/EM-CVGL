
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def save_sim_diff(sim_mat, qids, gids):
    
    match_mask = qids.unsqueeze(1) == gids
    true_pos = sim_mat[match_mask]

    top_idx = sim_mat.argsort(dim=-1, descending=True)[:,0]
    match_mask2 = torch.stack([qids != gids[top_idx], qids == gids[top_idx]], dim=-1)
    top2_sim = sim_mat.sort(dim=-1, descending=True)[0][:,:2]
    hard_neg = top2_sim[match_mask2]

    diff = true_pos - hard_neg

    torch.save(diff, "IM001_diff.t")