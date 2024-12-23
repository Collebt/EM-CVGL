import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN1
import torch.nn.functional as F

import numpy as np
from utils.utils import MODEL
from torch.optim import *

   
@MODEL.register
class AutoEnc2_share(torch.nn.Module):
    
    def __init__(self, out_dim=2048, vec_dim=2048, GNN='spline', num_steps=1,  
                 norm=True, normvec=True, residul=False, identify=False, query_num=701, 
                 temperature_init=14.28,
                 bias_init=-10,
                 **opt):
        super(AutoEnc2_share, self).__init__()
        self.num_steps = num_steps

        hidden = out_dim
        self.enc1 = Seq(
            Lin(out_dim, vec_dim),
            BN1(vec_dim)
        )
        self.dec1 = Seq(
            Lin(vec_dim, out_dim),
            BN1(out_dim)
        )

        self.enc2 = self.enc1 # share encoder
        self.dec2 = self.dec1 # share decoder
        
        self.res = residul
        self.identify = identify
        self.query_num = query_num

        temperature = torch.nn.Parameter(torch.ones([]) * np.log(temperature_init))
        bias = torch.nn.Parameter(torch.ones([]) * bias_init)
        self.logit_scale = {"t":temperature, "b":bias}
        
        self.pool_layer = nn.AdaptiveAvgPool1d(1)
        self.l2norm = lambda x: F.normalize(x, dim=-1) if norm else lambda s: s
        self.l2normvec = lambda x: F.normalize(x, dim=-1) if normvec else lambda x: x

    def run_train(self, data):
        src_x = data['x_s'] # satellite
        # if self.dro_num > 1: # multi drone images  [b, n, c]
        if data['x_t'].dim() == 3:
            tgt_x = data['x_t']
            b, n, c = tgt_x.shape
            tgt_x = tgt_x.reshape(b*n, c)[:self.query_num] 
        else:
            # drone
            tgt_x = data['x_t'][:self.query_num] 
        data['x_t'] = tgt_x

        enc_a = self.enc1(src_x)
        enc_b = self.enc2(tgt_x)

        if self.res:
            enc_a = enc_a + src_x
            enc_b = enc_b + tgt_x


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

        data['norm_src'] = self.l2norm(src_x)
        data['norm_tgt'] = self.l2norm(tgt_x)

        data['enc_b'] = self.l2norm(enc_b)
        data['enc_a'] = self.l2norm(enc_a)

        data['enc_ba'] = self.l2norm(enc_ba)
        return data

    def run_eval(self, data):
        src = data['x'] 
        # DEBUG###############
        if self.identify:
            enc_x = src
            fake_src = src 
        ################
            
        elif data['mode'][0] == 'sat': # satellite view
            enc_x = self.enc1(src)
            if self.res:
                enc_x = enc_x + src
            fake_src = self.dec1(enc_x)
        else: # drone view
            enc_x = self.enc2(src)
            if self.res:
                enc_x = enc_x + src
            fake_src = self.dec2(enc_x)

        data['out'] = self.l2norm(enc_x)
        data['vec'] = self.l2normvec(fake_src)

        # data['logit_scale'] = self.logit_scale
        return data


    def forward(self, data):       
        if self.training:
            data = self.run_train(data)
        else:
            data = self.run_eval(data)
        return data 
        

    def build_opt(self, opt):
        optimizer_a = eval(opt.train.optim)([
            {'params': self.enc1.parameters(), 'lr': opt.train.lr},
            {'params': self.dec1.parameters(), 'lr': opt.train.lr},
            {'params': self.logit_scale['t'], 'lr': opt.train.lr},
            {'params': self.logit_scale['b'], 'lr': opt.train.lr}])

        return optimizer_a 

