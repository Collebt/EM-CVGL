import torch
import matplotlib.pyplot as plt
from utils.utils import LOSS
from torch import nn


def filter_InfoNCE(sim_mat, sim_mat2, logit_scale, loss_fn, label1, label2):

    logits_per_image1 = logit_scale * sim_mat
    logits_per_image2 = logit_scale * sim_mat2
    loss = (loss_fn(logits_per_image1, label1) + loss_fn(logits_per_image2, label2))/2
    
    return loss



@LOSS.register
class CycleAELoss(nn.Module):

    def __init__(self, weight, args, recorder, device):
        super().__init__()

        self.loss_function = nn.CrossEntropyLoss(label_smoothing=args.train.label_smoothing)
        self.recon_fn = torch.nn.L1Loss()
        self.cycle_fn = torch.nn.L1Loss()
        self.pseudo_fn = None
        self.fea_cyc_fn = torch.nn.MSELoss()

        self.device = device
        self.w = weight
        self.recorder = recorder
        self.dro_num = args.train.dro_num
        self.lambda_A = args.train.lambda_A
        self.lambda_B = args.train.lambda_B
        self.idt_w = args.train.idt_w
        self.feat_w = args.train.feat_w

        self.thr = args.train.pseudo_thr
        self.mutual_match = args.train.mutual_match
        self.keep_neg = args.train.keep_neg

    def forward(self, data, logit_scale, args=None):

        fake_AA = data['fake_AA']
        fake_BB = data['fake_BB']

        enc_b = data['enc_b'] 
        enc_a = data['enc_a'] 

        real_A = data['x_s']
        real_B = data['x_t'].squeeze()

        A_id = data['y_s']
        B_id = data['y_t']
        
        

        sim_mat = torch.einsum('md, nd-> mn', enc_a, enc_b)
        m,n = sim_mat.shape
        sim_mat_multi = sim_mat.reshape(m, n//self.dro_num, self.dro_num)
        sim_mat_mean = sim_mat_multi.mean(-1)

        score,  retri_idx = sim_mat_mean.max(-1)
        score_T, retri_idx_T = sim_mat_mean.T.max(-1)


        # reconstruction loss
        loss_recon_A = self.idt_w * self.recon_fn(fake_AA, real_A)
        loss_recon_B = self.idt_w * self.recon_fn(fake_BB, real_B)

        # GT pair label only for evaluation during training
        thres = self.thr 
        gt_mask = A_id.unsqueeze(1) == B_id[:n].unsqueeze(0) # for real match accuracy
        
        
        # mutual matching for pseudo-label generation
        mutual_mask = (retri_idx_T[retri_idx] == torch.arange(0, len(retri_idx)).to(retri_idx))
        mutual_mask_T = (retri_idx[retri_idx_T] == torch.arange(0, len(retri_idx_T)).to(retri_idx_T))
        if  self.mutual_match and  mutual_mask.sum()>10:
            # retri_idx
            sel_mat =  sim_mat_mean[mutual_mask]
            
            if self.keep_neg:
                sel_mat_T = (sim_mat_mean.T)[mutual_mask_T]
                gt_mask = gt_mask[mutual_mask]
            else:
                sel_mat_T = (sel_mat.T)[mutual_mask_T]
                sel_mat = sel_mat_T.T
                gt_mask = (gt_mask[mutual_mask].T)[mutual_mask_T].T

            score,  sel_idx = sel_mat.max(-1)
            score_T, sel_idx_T = sel_mat_T.max(-1)

        else:
            sel_mat, sel_mat_T = sim_mat_mean, sim_mat_mean.T
            sel_idx, sel_idx_T = retri_idx, retri_idx_T


        sel_mat, labels1 = sel_mat[score > thres], sel_idx[score > thres]
        sel_mat_T, labels2 = sel_mat_T[score_T > thres], sel_idx_T[score_T > thres]


        logit_scale_exp = logit_scale["t"].exp()
        loss_pseudo = filter_InfoNCE(sel_mat, sel_mat_T, logit_scale_exp, self.loss_function, labels1, labels2)

        
        # for evaluation during training
        _, gt_idx = gt_mask.max(-1)

        real_acc = (sel_idx == gt_idx.to(sel_idx)).sum() / torch.tensor(len(sel_mat)).to(torch.float)

        self.recorder.update('Lrec_A', loss_recon_A.item(), args.train.batch_size, type='f')
        self.recorder.update('Lrec_B', loss_recon_B.item(), args.train.batch_size, type='f')

        self.recorder.update('Lpseudo', loss_pseudo.item(), args.train.batch_size, type='f')
        self.recorder.update('real_acc', real_acc.item(), args.train.batch_size, type='%')

        final_loss = loss_pseudo + loss_recon_A + loss_recon_B #+ loss_cross_recon_A + loss_cross_recon_B
        
        return final_loss



def normalize(x, dim=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, dim, keepdim=True).expand_as(x) + 1e-6)
    return x

def compute_dist(opt_feat_embed, sar_feat_embed, squred=True):
    # get the distance matrix between optical features and sar features
    m, n = opt_feat_embed.shape[0], sar_feat_embed.shape[0]

    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    emb1_pow = torch.pow(opt_feat_embed, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(sar_feat_embed, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # mat_sum = torch.einsum('md,nd->mn', opt_feat_embed,sar_feat_embed )
    # dist_mat = dist_mat.addmm(1, -2, opt_feat_embed, sar_feat_embed.t()) #ori admm
    dist_mat = emb1_pow + emb2_pow - 2*(opt_feat_embed @ sar_feat_embed.t())

    dist_mat[dist_mat < 0] = 0
    if not squred:
        dist_mat = dist_mat.clamp(min=1e-12).sqrt()
    return dist_mat


def compute_dist_my(src_feat, tgt_feat, squred=True):
    """
    L2 distance between two features,
    
    """
    # get the distance matrix between optical features and sar features
    m, n, d = src_feat.shape[0], tgt_feat.shape[0], src_feat.shape[-1]

    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    emb1_pow = src_feat.unsqueeze(1).expand(m, n, d)
    emb2_pow = tgt_feat.unsqueeze(0).expand(m, n, d)
    dist_mat = (emb1_pow - emb2_pow).pow(2).sum(-1)
    if not squred:
        dist_mat = dist_mat.clamp(min=1e-12).sqrt()
    return dist_mat
