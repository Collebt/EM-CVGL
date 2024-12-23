import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN1
import torch.nn.functional as F

import numpy as np
from utils.utils import MODEL
from torch.optim import *

   
@MODEL.register
class AutoEnc(torch.nn.Module):
    
    def __init__(self, out_dim=2048, vec_dim=2048, GNN='spline', num_steps=1,  norm=True, normvec=True, **opt):
        super(AutoEnc, self).__init__()

        self.num_steps = num_steps

        self.enc1 = Seq(
            Lin(out_dim, vec_dim),
            BN1(vec_dim)
        )
        self.enc2 = Seq(
            Lin(out_dim, vec_dim),
            BN1(vec_dim)
        )
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pool_layer = nn.AdaptiveAvgPool1d(1)
        self.l2norm = lambda x: F.normalize(x, dim=-1) if norm else lambda s: s
        self.l2normvec = lambda x: F.normalize(x, dim=-1) if normvec else lambda x: x

        # self.l2norm = lambda x:x

    def forward(self, data):       
        if self.training:
            src_x = data['x_s'] # satellite
            tgt_x = data['x_t'] # drone
            
            # if self.dro_num > 1: # multi drone images  [b, n, c]
            if tgt_x.dim() == 3:
                b, n, c = tgt_x.shape
                tgt_x = tgt_x.reshape(b*n, c)

            src_tgt = self.enc1(src_x)
            tgt_src = self.enc1(tgt_x)

            # src_tgt = src_x
            # tgt_src = tgt_x


            src_tgt_src = self.enc2(src_tgt)
            tgt_src_tgt = self.enc1(tgt_src)

            # if self.dro_num > 1: # multi drone images  [b, n, c]
            #     tgt_src = tgt_src.reshape(b, n, -1)
            #     tgt_src_tgt = tgt_src_tgt.reshape(b, n, -1)

            data['out_s'] = self.l2norm(src_tgt)
            data['out_t'] = self.l2norm(tgt_src)

            data['vec_s'] = self.l2normvec(src_tgt_src)
            data['vec_t'] = self.l2normvec(tgt_src_tgt)
            # data['logit_scale'] = self.logit_scale

        else:
            src = data['x'] 
            if data['mode'][0] == 'sat': # satellite view
                src_tgt = self.enc1(src)
                src_tgt_src = self.enc2(src_tgt)
            else: # drone view
                src_tgt = self.enc1(src)
                src_tgt_src = self.enc2(src_tgt)


            # # DEBUG###############
            # src_tgt = src
            # src_tgt_src = src 
            # ################
            
            # src_tgt = enc(src)
            # src_tgt_src = dec(src_tgt)
            # src_tgt = src

            data['out'] = self.l2norm(src_tgt)
            data['vec'] = self.l2normvec(src_tgt_src)

            # data['logit_scale'] = self.logit_scale

        return data 
        

    def build_opt(self, opt):
        optimizer = eval(opt.train.optim)([
            {'params': self.enc1.parameters(), 'lr': opt.train.lr},
            {'params': self.enc2.parameters(), 'lr': opt.train.lr}])
        return optimizer

