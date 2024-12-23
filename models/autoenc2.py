import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN1
import torch.nn.functional as F

import numpy as np
from utils.utils import MODEL
from torch.optim import *

   
@MODEL.register
class AutoEnc2(torch.nn.Module):
    
    def __init__(self, out_dim=2048, vec_dim=2048, GNN='spline', num_steps=1,  norm=True, normvec=True, **opt):
        super(AutoEnc2, self).__init__()

        self.num_steps = num_steps

        self.enc1 = Seq(
            Lin(out_dim, vec_dim),
            BN1(vec_dim)
        )
        self.enc2 = Seq(
            Lin(out_dim, vec_dim),
            BN1(vec_dim)
        )

        self.dec1 = Seq(
            Lin(vec_dim, out_dim),
            BN1(out_dim)
        )

        self.dec2 = Seq(
            Lin(vec_dim, out_dim),
            BN1(out_dim)
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

               

            enc_a = self.enc1(src_x)
            enc_b = self.enc2(tgt_x)

            fake_AA = self.dec1(enc_a)
            fake_AB = self.dec2(enc_a)

            fake_BA = self.dec1(enc_b)
            fake_BB = self.dec2(enc_b)

            enc_ab = self.enc2(fake_AB)
            fake_ABA = self.dec1(enc_ab)

            enc_ba = self.enc1(fake_BA)
            fake_BAB = self.dec2(enc_ba)


            data['fake_AA'] = self.l2norm(fake_AA)
            data['fake_BB'] = self.l2norm(fake_BB)

            data['fake_ABA'] = self.l2norm(fake_ABA)
            data['fake_BAB'] = self.l2norm(fake_BAB)

            data['enc_b'] = self.l2norm(enc_b)
            data['enc_a'] = self.l2norm(enc_a)

            data['enc_ba'] = self.l2norm(enc_ba)


        else:
            src = data['x'] 
            if data['mode'][0] == 'sat': # satellite view
                enc_x = self.enc1(src)
                fake_src = self.dec1(enc_x)
            else: # drone view
                enc_x = self.enc2(src)
                fake_src = self.dec2(enc_x)

            # # DEBUG###############
            # src_tgt = src
            # src_tgt_src = src 
            # ################
            
            # src_tgt = enc(src)
            # src_tgt_src = dec(src_tgt)
            # src_tgt = src

            data['out'] = self.l2norm(enc_x)
            data['vec'] = self.l2normvec(fake_src)

            # data['logit_scale'] = self.logit_scale

        return data 
        

    def build_opt(self, opt):
        optimizer_a = eval(opt.train.optim)([
            {'params': self.enc1.parameters(), 'lr': opt.train.lr},
            {'params': self.dec1.parameters(), 'lr': opt.train.lr}])
        optimizer_b = eval(opt.train.optim)([
            {'params': self.enc2.parameters(), 'lr': opt.train.lr},
            {'params': self.dec2.parameters(), 'lr': opt.train.lr}])
        
        return optimizer_a, optimizer_b

