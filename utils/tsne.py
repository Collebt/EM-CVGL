from numpy import float16
from sklearn.manifold import TSNE
import torch as tr
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def feat_tsne_pair(x:Tensor, y:Tensor, match_mat:Tensor, name='tsne'):
    """
    x: [ns, dim] 
    y: [nt, dim]
    match_mat : {0,1} matching matrix
    name: the output png file name

    return savefig in maindir
    """
    ns = x.shape[0]
    indes = tr.where(match_mat==1)
    input = tr.cat([x, y], dim=0).numpy()
    output = tr.tensor(TSNE( n_components=2, learning_rate=100 ,init='random').fit_transform(input))
    x_embed, y_embed = output[:ns], output[ns:] # return the pair
    plt.scatter(output[:,0], output[:,1], c='b')
    for row, col in zip(*indes):
        X = [x_embed[row,0].item(),y_embed[col,0].item()]
        Y = [x_embed[row,1].item(),y_embed[col,1].item()]
        plt.scatter(X,Y)
    plt.savefig(name)
    plt.close()


def eval_tsne(q_feat, g_feat, qid, gid, f_path, select_scene=20, select_drone=5, r=200):
        # get matched pair
        ns = q_feat.shape[0]
        qid_idx, gid_idx = torch.where(qid.unsqueeze(-1) == gid)
        nice_id = torch.load('nice_scene.t')
        nice_qmask = (qid.unsqueeze(-1) == nice_id.to(qid)).sum(-1) == 1
        nice_gmask = (gid.unsqueeze(-1) == nice_id.to(gid)).sum(-1) == 1

        sim_mat = q_feat @ g_feat.T
        true_g_feat = g_feat[gid_idx]
        gt_sim = (q_feat * true_g_feat).sum(-1)

        top_indices = torch.argsort(sim_mat, dim=-1, descending=True)[:,0]
        tp_mask, fp_mask = top_indices==gid_idx, top_indices!=gid_idx

        # feat_dict = {'q_feat':q_feat, "g_feat": g_feat, "qid":qid, "gid":gid }
        # torch.save(feat_dict, 'adapter_feat.t')
        

        matched_qids = qid_idx[top_indices==gid_idx]
        unmatch_qids = qid_idx[top_indices!=gid_idx]

        tp_gids = gid_idx[tp_mask]
        fp_gids = gid_idx[fp_mask]

        tp_statistic = torch.zeros(qid.max()+1)
        fp_statistic = torch.zeros(qid.max()+1)
        for idx in qid[tp_mask]:  
            tp_statistic[idx] += 1
        for idx in qid[fp_mask]:  
            fp_statistic[idx] += 1 

        tp_sim = gt_sim[tp_mask]
        fp_sim = gt_sim[fp_mask]

        torch.save({'tp': tp_statistic, 'fp': fp_statistic, 'tp_sim': tp_sim, 'fp_sim': fp_sim}, 'custom_pair.t')



        # TSNE
        input = torch.cat([q_feat[nice_qmask], g_feat[nice_gmask]], dim=0)
        output = torch.tensor(TSNE(n_components=2, learning_rate=100 ,init='random').fit_transform(input.cpu()))
        


        x_embed, y_embed = output[:5400], output[5400:] # return the pair
        x_embed, y_embed = output[:ns], output[ns:]

        ################################# matplot ###########################
        # scatter points
        plt.scatter(x_embed[:,0], x_embed[:,1], c='lightskyblue', alpha=0.5, marker='.', s=0.1)
        plt.scatter(y_embed[:,0], y_embed[:,1], c='lightcoral', alpha=0.5, marker='o', s=2, edgecolors=None)
        plt.savefig(os.path.join(f_path, 'tsne_base.png'))
        plt.close()
        
        # show gt_link
        # tp_x_emb, tp_y_emb = x_embed[tp_mask], y_embed[tp_gids]
        # X = torch.stack([tp_x_emb[:,0] , tp_y_emb[:,0]], dim=-1)
        # Y = torch.stack([tp_x_emb[:,1] , tp_y_emb[:,1]], dim=-1)
        # for x, y in zip(X, Y):
        #     plt.plot(x,y, c='k')
        
        # show matched pair tsne
        plt.savefig(os.path.join(f_path, 'tsne_base.png'))
        plt.close()
        print(f'TSNE result is shown in {f_path}')

        #############################plotly ##########################
        #interactive ploty scatter
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_embed[:,0], y=x_embed[:,1],
            name='Drone feature',
            marker=dict(color='rgba(156, 165, 196, 0.95)',line_color='rgba(156, 165, 196, 1.0)',
        )))
        fig.add_trace(go.Scatter(
            x=y_embed[:,0], y=y_embed[:,1],
            name='Satellite feature',
            marker=dict(color='rgba(204, 204, 204, 0.7)',line_color='rgba(217, 217, 217, 1.0)',
        )))
        fig.update_traces(mode='markers', marker=dict(line_width=1, symbol='circle', size=16))
        color_ind = gt_sim
        color_ind[color_ind<0.9] = 0.4
        color_ind[(color_ind>0.9) * (color_ind<0.95)] = 0.7
        color_ind[(color_ind>0.95)] = 1
        
        #match pairs
        for pair_idx, (xid, yid) in enumerate(zip(matched_qids, matched_gids)):
            X = [x_embed[xid,0].item(),y_embed[yid,0].item()]
            Y = [x_embed[xid,1].item(),y_embed[yid,1].item()]
            fig.add_trace(go.Scatter(x=X,y=Y, mode='lines',
                                    line=dict(color=f'rgba(0,100,80,{color_ind[xid]})', width=float(6*color_ind[xid])),
                                    showlegend=False
                                    )          
            )
        #false match pairs
        for pair_idx, (xid, yid) in enumerate(zip(unmatch_qids, unmatch_gids)):
            X = [x_embed[xid,0].item(),y_embed[yid,0].item()]
            Y = [x_embed[xid,1].item(),y_embed[yid,1].item()]
            fig.add_trace(go.Scatter(x=X,y=Y, mode='lines',
                                    line=dict(color=f'rgba(180,180,180,{color_ind[xid]})', width=float(6*color_ind[xid])),
                                    showlegend=False
                                    )          
            )
        #unmatch pairs
        for pair_idx, (xid, yid) in enumerate(zip(unmatch_qids, false_gids)):
            X = [x_embed[xid,0].item(),y_embed[yid,0].item()]
            Y = [x_embed[xid,1].item(),y_embed[yid,1].item()]
            fig.add_trace(go.Scatter(x=X,y=Y, mode='lines',
                                    line=dict(color=f'rgba(100,0,0,{color_ind[xid]})', width=float(6*color_ind[xid])),
                                    showlegend=False
                                    )          
            )
        fig.write_html(os.path.join(f_path, 'all_tsne.html'))

        # ## PCA
        # print('PCA is visualizings')
        # output = torch.tensor(PCA(n_components=2).fit_transform(input.cpu()))
        # x_embed, y_embed = output[:ns], output[ns:]
        # #interactive ploty scatter
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(
        #     x=x_embed[:,0], y=x_embed[:,1],
        #     name='SAR image feature',
        #     marker=dict(color='rgba(156, 165, 196, 0.95)',line_color='rgba(156, 165, 196, 1.0)',
        # )))
        # fig.add_trace(go.Scatter(
        #     x=y_embed[:,0], y=y_embed[:,1],
        #     name='RGB image feature',
        #     marker=dict(color='rgba(204, 204, 204, 0.7)',line_color='rgba(217, 217, 217, 1.0)',
        # )))
        # fig.update_traces(mode='markers', marker=dict(line_width=1, symbol='circle', size=16))
        # color_ind = inner_product_pair
        # color_ind[color_ind<0.9] = 0.4
        # color_ind[(color_ind>0.9) * (color_ind<0.95)] = 0.7
        # color_ind[(color_ind>0.95)] = 1
        
        # for pair_idx, (xid, yid) in enumerate(zip(matched_qids, matched_gids)):
        #     X = [x_embed[xid,0].item(),y_embed[yid,0].item()]
        #     Y = [x_embed[xid,1].item(),y_embed[yid,1].item()]
        #     fig.add_trace(go.Scatter(x=X,y=Y, mode='lines',
        #                             line=dict(color=f'rgba(0,100,80,{color_ind[xid]})', width=float(6*color_ind[xid])),
        #                             showlegend=False
        #                             )          
        #     )
        # #false match pairs
        # for pair_idx, (xid, yid) in enumerate(zip(unmatch_qids, unmatch_gids)):
        #     X = [x_embed[xid,0].item(),y_embed[yid,0].item()]
        #     Y = [x_embed[xid,1].item(),y_embed[yid,1].item()]
        #     fig.add_trace(go.Scatter(x=X,y=Y, mode='lines',
        #                             line=dict(color=f'rgba(180,180,180,{color_ind[xid]})', width=float(6*color_ind[xid])),
        #                             showlegend=False
        #                             )          
        #     )

        # #unmatch pairs
        # for pair_idx, (xid, yid) in enumerate(zip(unmatch_qids, false_gids)):
        #     X = [x_embed[xid,0].item(),y_embed[yid,0].item()]
        #     Y = [x_embed[xid,1].item(),y_embed[yid,1].item()]
        #     fig.add_trace(go.Scatter(x=X,y=Y, mode='lines',
        #                             line=dict(color=f'rgba(100,0,0,{color_ind[xid]})', width=float(6*color_ind[xid])),
        #                             showlegend=False
        #                             )          
        #     )
        # fig.write_html(f_path/'all_pca.html')
        # print(f'PCA result is shown in {f_path}')

    








    