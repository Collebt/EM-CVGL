import argparse
import sys
import torch
import torch.nn as nn
import torch.utils.data as data
from utils.visual import vis_retrieval_imgs
# try:
#     from torch_geometric.loader import DataLoader
# except:
from torch.utils.data import DataLoader
from data.dataset import * 

from utils.utils import *
from utils.save_dist_diff import save_sim_diff

import time
import random
import os.path as osp

import matplotlib.pyplot as plt
from tqdm import tqdm
from models import *
from loss import compute_dist

# from utils.tsne import eval_tsne

HOME = osp.expanduser('~')

def save_feat(gall_feat, gid, gall_name, view, savedir):

    if not os.path.exists(savedir):
        os.mkdir(savedir)
    torch.save(gall_feat, osp.join(savedir, f'{view}_feat'))
    torch.save(gid, osp.join(savedir, f'{view}_id'))
    torch.save(gall_name, osp.join(savedir, f'{view}_name'))
    print(f'feature are saved in: {savedir}')

class retrieval_test:
    def __init__(self, opt, query_loader, gall_loader, device='cuda', vis=False, S2D=False):
        #####################build reference image-base ######
        #init the gallery dataloader
        self.batch_size = opt.eval.batch_size
        self.query_loader = query_loader
        self.gall_loader = gall_loader

        #here the test index is not started from 0
        self.opt = opt
        self.vis = vis
        # self.remove_junk = opt.eval.remove_junk


        self.refine_batch = opt.eval.batch_size
        self.device = device
        self.S2D = S2D

    def extract_feat(self, model, dataloader, mode='sat'):
        """extract feature of reference images
        Returns:
            ref_fear: n-dimension extracted features of the reference images
            ref_id: the id of the location
            ref_pos: GPS position of the correspondent reference images
        """

        print(f'Extracting {mode} feature')
        model.eval()   #evaluate the network
        feat, feat_vec = [], []
        ids = []
        img_name = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                data = move_to(data, self.device)
                # to_cuda(data) if self.device == 'cuda' else data
                data = model(data)

                # collect feature
                feat.append(data['out'])
                feat_vec.append(data['vec'])
                ids.append(data['y'])
                img_name.extend(data['name'])

        emb_feat = torch.cat(feat, dim=0)
        emb_vec = torch.cat(feat_vec, dim=0)
        target_id = torch.cat(ids, dim=0)


        out_feat = emb_feat if self.opt.eval.output == 'feat' else emb_vec
        return out_feat, target_id, img_name


    def eval_retrieval(self, sim_mat, qid, gid):
        # compute distmat
        mAP = []
        retrieval_idx = torch.argsort(sim_mat, dim=1, descending=True)
        pred_qid = gid[retrieval_idx]
        matches = (pred_qid == qid.reshape(-1, 1)) #get binary matching mat along the reference
        hit_pos = torch.where(matches == 1)[1]

        cmc = matches.sum(0).cumsum(-1) 
        cmc = cmc.to(torch.float) / qid.shape[0]
 
        mAP = torch.mean(1 / (hit_pos.to(torch.float) + 1))
        return cmc, mAP, matches

    def evaluation(self, model, savedir=None):
        gall_feat, gid, gall_name  = self.extract_feat(model, self.gall_loader, mode='sat')
        if savedir:
            save_feat(gall_feat, gid, gall_name, 'sat', savedir)
        query_feat, qid, query_name = self.extract_feat(model, self.query_loader, mode='dro')
        if savedir:
            save_feat(query_feat, qid, query_name, 'dro', savedir)

        # evaluate retrieval accuracy
       

        gal_num = gid.shape[0]

        print(f'-----------------------')
        print(f'dim:{query_feat.shape[-1]}|#Ids\t| #Img ')
        print(f'-----------------------')
        print(f'Query\t|{torch.unique(qid).shape[0]}\t|{query_feat.shape[0]} ')
        print(f'Gallery\t|{gal_num}\t|{gal_num}')
        print(f'-----------------------')


        ### visualizartion ############
        # eval_tsne(query_feat, gall_feat, qid, gid, f_path='outputs')
        vis_path = 'outputs'
        sim_mat = query_feat @ gall_feat.T
        if self.S2D:
            gal_img_path = osp.join(HOME, "data/University-Release/test/gallery_drone")
            que_img_path = osp.join(HOME, "data/University-Release/test/query_satellite")
            gall_name = torch.load(osp.join(HOME, "data/University-Release/test/anyloc_feat_s2d/dro_name"))
            query_name = torch.load(osp.join(HOME, "data/University-Release/test/anyloc_feat_s2d/sat_name"))
        else:
            que_img_path = osp.join(HOME, "data/University-Release/test/drone")
            gal_img_path = osp.join(HOME, "data/University-Release/test/satellite")
            gall_name = torch.load(osp.join(HOME, "data/University-Release/test/anyloc_feat/sat_name"))
            query_name = torch.load(osp.join(HOME, "data/University-Release/test/anyloc_feat/dro_name"))

        gall_name = [osp.join(gal_img_path, p) for p in gall_name]
        query_name = [osp.join(que_img_path, p) for p in query_name]

        # save_sim_diff(sim_mat, qid, gid)

        if self.vis:
            base_good_idx = vis_retrieval_imgs(sim_mat, 
                                            qid, 
                                            gid,
                                            query_name, 
                                            gall_name,
                                            log_path=vis_path, 
                                            fix_id= [29, 692, 408, 652], #[777,7362,36785,11786], 
                                            f_name='s2d_vis_adp.pdf') #show retrieval results
        

        ############### top k  ######################
        gl = gid.cpu().numpy()
        ql = qid.cpu().numpy()
        
        print("Compute Scores:")
        CMC = torch.IntTensor(len(gid)).zero_()
        ap = 0.0
        for i in tqdm(range(len(qid))):
            ap_tmp, CMC_tmp = eval_query(query_feat[i], ql[i], gall_feat, gl)
            if CMC_tmp[0]==-1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
        
        AP = ap/len(qid) #*100
        
        CMC = CMC.float()
        CMC = CMC/len(qid) #average CMC

        top1p = len(gid)//100
        

        ######################## top 1% ###################
        # top1 = round(len(gid)*0.01)
        # string = []
        # for i in [1,5,10]:
        #     string.append(f'Recall@{i}: {CMC[i-1]*100:.4f}')
            
        # string.append(f'Recall@top1: {CMC[top1]*100:.4f}')
        # string.append(f'AP: {AP:.4f}')  

        # print(' - '.join(string)) 

        ###################### my #########################
        # similarity = - compute_dist(query_feat, gall_feat)
        # CMC, AP, coarse_match = self.eval_retrieval(similarity, qid, gid)
        # print(f'CMC: {CMC}')
        print(f'Retrieval: top-1:{CMC[0]:.2%} | top-5:{CMC[4]:.2%} | top-10:{CMC[9]:.2%} | top-1%:{CMC[top1p]:.2%} | AP:{AP:.2%}')

def eval_query(qf,ql,gf,gl):

    score = gf @ qf.unsqueeze(-1)
    
    score = score.squeeze().cpu().numpy()

    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]    

    # good index
    query_index = np.argwhere(gl==ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl==-1)
    
    
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch Cross-View Testing for RK-vanilla')
    parser.add_argument('cfg', default='configs/split.yml', type=str, help='yaml config path')
    parser.add_argument('--param_path', '-p',  type=str, help='the relative path of model parameters for testing')
    parser.add_argument('--gpu', '-g', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--extend', action='store_true', 
                        help='use the extend 160k satellite gallery')
    parser.add_argument('--S2D', action='store_true', 
                        help='satellite -> drone')
    parser.add_argument('--save_dir', type=str,
                        help='dir to save the output features tensor')
    
    

    args = parser.parse_args()
    opt = update_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)  # 为所有GPU设置随机种子
    np.random.seed(1)
    random.seed(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # set data path
    load_param = opt.param_path if opt.param_path is not None else None
    log_path = osp.dirname(opt.param_path) if opt.param_path is not None else osp.join('outputs', opt.cfg.split('/')[-1].split('.')[0])

    if opt.save_dir is not None:
        savedir = osp.join(HOME, opt.save_dir)
        if not osp.exists(savedir):
            os.mkdir(savedir)
    else:
        savedir = None


    ## set log output
    sys.stdout = Logger(osp.join(log_path, f'Test_{opt.eval.dataset}_{time.asctime()}.log'))
    print(f"==========\nArgs:{opt}\n==========")

    print('==> Building model..')
    model = MODEL[opt.model.name](**opt.model)


    # model.load_state_dict(torch.load(load_param)) # for pth 
    if load_param is not None:
        if os.path.isfile(load_param):  
            checkpoint = torch.load(load_param)
            if opt.model.name == 'ViT_base_pretrain':
                model.transformer.load_param(str(load_param))
            elif 'model' in checkpoint:
                model_param  = checkpoint['model']
                model.load_state_dict(model_param)
                # model.transformer.load_param(str(load_param)) #adapt the vit
            else:
                model.load_state_dict(checkpoint)
            print(f'==> loaded checkpoint from {load_param}')
        else:
            print(f'==> no checkpoint found at {load_param}')
    # model.classifier.classifier = nn.Sequential()
    model.to(device)
   
    sat_mode = 'sat_160k' if args.extend else 'sat'
    queryset = DATASET[opt.eval.dataset](mode='dro', **opt.eval)
    gallset = DATASET[opt.eval.dataset](mode=sat_mode, **opt.eval)
    
    # for S2D
    if opt.S2D:
        gallset = DATASET[opt.eval.dataset](mode='dro', **opt.eval)
        queryset = DATASET[opt.eval.dataset](mode=sat_mode, **opt.eval)


    query_loader = DataLoader(queryset, batch_size=opt.eval.batch_size, shuffle=False, num_workers=opt.workers)
    gall_loader = DataLoader(gallset, batch_size=opt.eval.batch_size, shuffle=False, num_workers=opt.workers)

    result_test = retrieval_test(opt, query_loader, gall_loader, device=device, S2D=opt.S2D)
    result_test.evaluation(model, savedir)
    print('test finished')


    