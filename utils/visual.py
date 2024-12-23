
import torch
import matplotlib
# matplotlib.use('Agg')
import os

import matplotlib.pyplot as plt

def wd_vis(record_dict, *args):
    """visualize the w distance of the features

    Args:
        record_dict(dict)--wd(dict)--  name(str): output file name
                                    |- wd_list(list): list of the output 1-d values
                                    
        [src_feat (tensor): w-value of the source feature output from Discriminator. [b]
        tgt_feat (tensor): w-value of the target feature output from Discriminator. [b]]
    """
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()

    f_name = record_dict['wd']['name']
    f_name = record_dict['wd']['name']
    for i, feats in enumerate(record_dict['wd']['feat']):
        src_feat = feats[0].squeeze().detach().cpu()
        tgt_feat = feats[1].squeeze().detach().cpu()

        src_label = torch.ones_like(src_feat)
        tgt_label = torch.zeros_like(tgt_feat)
        src_pos = torch.stack([src_feat, src_label])
        tgt_pos = torch.stack([tgt_feat, tgt_label])
        axs[i].scatter(src_pos[0], src_pos[1], alpha=0.2, c='blue', label='SAR')
        axs[i].scatter(tgt_pos[0], tgt_pos[1], alpha=0.2, c='red', label='optical')
        axs[i].legend()
        for src_p, tgt_p in zip(src_pos.T, tgt_pos.T):
            X = [src_p[0], tgt_p[0]]
            Y = [src_p[1], tgt_p[1]]
            axs[i].plot(X, Y, c='black', alpha=0.22, linewidth=0.5)
        axs[i].set_yticks([]) 
        axs[i].set_xlabel(f'epoch{i*171}')
        

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f_name, dpi=200)
    plt.close()


def cls_vis(record_dict, *args):
    """visualize the class result of the features

    Args:
        record_dict(dict)--wd(dict)--  name(str): output file name
                                    |- feat(list): list of the output 1-d values
                                    
        [src_feat (tensor): classifcation result of the source feature output from Discriminator. [b]
        tgt_feat (tensor): classifcation result of the target feature output from Discriminator. [b]
    """

    f_name = record_dict['wd']['name']

    src_pos = record_dict['wd']['feat'][0].T.squeeze().detach().cpu() #shape =[2, b]
    tgt_pos = record_dict['wd']['feat'][1].T.squeeze().detach().cpu()


    plt.scatter(src_pos[0], src_pos[1], alpha=0.2, c='blue', label='SAR')
    plt.scatter(tgt_pos[0], tgt_pos[1], alpha=0.2, c='red', label='optical')
    plt.legend()
    for src_p, tgt_p in zip(src_pos.T, tgt_pos.T):
        X = [src_p[0], tgt_p[0]]
        Y = [src_p[1], tgt_p[1]]
        plt.plot(X, Y, c='black', alpha=0.22, linewidth=0.5)
    plt.xlabel(f'optical')
    plt.ylabel(f'SAR')   
    plt.savefig(f_name, dpi=200)
    plt.close()

def tri_vis(record_dict, *args):
    """visualize the distance between positive& negative samples.
    Args:
        record_dict(dict)--tri_vis(dict)--  name(str): output file name
                                        |- feat(list): distance of the output 1-D samples
                                    
        [pos_distance (tensor): the positive distance of samples from a batch. [b]
        neg_distance (tensor): the negative distance of samples from a batch. [b]
    """
    batch_idx = args[0]
    if batch_idx < 900:
        return 0
    f_name = record_dict['tri_vis']['name']

    feat = torch.stack(record_dict['tri_vis']['feat']) #shape = [Nx2, 2, batch_size]
    iters = feat.shape[0] // 2

    feat = feat.permute(1, 0, 2) #shape = [2, 2N, batch_size]
    

    pos_d = feat[0].reshape(iters, -1).detach().cpu() #[N, batch_size]
    neg_d = feat[1].reshape(iters, -1).detach().cpu() #[N, batch_size]

    

    iters_x = torch.arange(0, iters).reshape(-1, 1).repeat(1, pos_d.shape[-1]).detach().cpu()


    plt.scatter(iters_x.flatten(), pos_d.flatten(), alpha=0.22, c='green', label='positive dist')
    plt.scatter(iters_x.flatten(), neg_d.flatten(), alpha=0.22, c='red', label='negative dist')
    plt.legend()

    pos_dis_avg = pos_d.mean(-1)
    neg_dis_avg = neg_d.mean(-1)

    plt.plot(torch.arange(0, iters), pos_dis_avg, c='green')
    plt.plot(torch.arange(0, iters), neg_dis_avg, c='red')

    plt.xlabel(f'iterations')
    plt.ylabel(f'distance')   
    plt.savefig(f_name, dpi=200)
    plt.close()

def eval_mse_statistic(self, feat_src, feat_tgt, gt_id, f_path, show_num=200):
        """statistic the mes of REAL MATHCED PARIS
        feat_src: [ns, d] query features,
        feat_tgt: [nt, d] gallery features.
        dismat:[ ns, nt] inner product distance
        """
        ns, nt = feat_src.shape[0], feat_tgt.shape[0]
        qid_idx, gid_idx = torch.where(gt_id.unsqueeze(-1) == self.ref_id)
        feat_src_pair = feat_src[qid_idx]
        feat_tgt_pair = feat_tgt[gid_idx]
        mse_pairs = torch.mean((feat_src_pair - feat_tgt_pair) ** 2, dim=-1)
        inner_product_pair = torch.sum(feat_src_pair * feat_tgt_pair, dim=-1)
        fig, axs = plt.subplots(1,2)
        axs[0].hist(mse_pairs.cpu(), bins=100)
        axs[1].hist(inner_product_pair.cpu(), bins=100)
        # plt.hist(mse_pairs)
        plt.savefig(f_path / 'statistic.png')
        plt.close()
        print(f'MSE result is shown in {f_path}')





def vis_retrieval_imgs(pred_retrieval, qids, gids, query_name, gall_name, show_num=4, gall_num=5, log_path=None, fix_id=None, S2D=False, f_name='base'):
        
        def draw_imgs(query_img, gall_img, f_name):
            fig, axs = plt.subplots(show_num, gall_num+1, figsize=(gall_num+1, show_num))
            
            for r in range(show_num):
                for c in range(gall_num+1):
                    if c == 0: # show query
                        img  = plt.imread(query_img[r])
                    else: # shwo gallery
                        img  = plt.imread(gall_img[r][c-1])
                        # axs[0, c].set_title('GT-Ref')
                        # axs[r, c].set_xlabel(f'rank:{gt_rank:.0f}')
                    axs[r, c].imshow(img)
                    axs[r, c].set_xticks([])
                    axs[r, c].set_yticks([]) 
                    axs[r, c].spines['top'].set_visible(False)
                    axs[r, c].spines['right'].set_visible(False)
                    axs[r, c].spines['bottom'].set_visible(False)
                    axs[r, c].spines['left'].set_visible(False)
                # axs[0, 0].set_title('Query')
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig(os.path.join(log_path, f_name))
            plt.close()
                 
                
        # find good rssult
        top5_idx = pred_retrieval.argsort(dim=-1, descending=True)[:,:gall_num]
        match_mask = qids.unsqueeze(1) == gids[top5_idx]



        if not os.path.exists(log_path):
            os.mkdir(log_path)
             
        # find the true dict
        if fix_id is None:
            match_num = match_mask.sum(-1)
            true_pos_idx = torch.where(match_num>0)[0]
            fix_id = true_pos_idx[:5]
            good_num = match_num[match_num>0]
            

            with open(os.path.join(log_path, 'true_idx_s2d_base.txt'), 'w') as f:
                for idx, gn in zip(true_pos_idx, good_num):
                    f.write(f'{idx.item()},{gn.item()}\n')
             

        # use fix id
        que_img = [query_name[i] for i in fix_id]
        gal_img = [[gall_name[j]  for j in  top5_idx[i, :5]]for i in fix_id]

        draw_imgs(que_img, gal_img, f_name)
        
        return None