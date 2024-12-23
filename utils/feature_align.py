import torch
from torch import Tensor
import matplotlib.pyplot as plt


def find_cluster_pts(im, pts_norm):
    """
    :param im: feature map [c, w, h]
    :param pts: :[n, 2]
    """
    with torch.no_grad():
        C, H, W = im.shape
        pts = pts_norm * torch.tensor([H, W]).to(pts_norm) 
        grid = pts.floor().to(dtype=int, device=im.device) #
        rcd = {}
        
        for i, (g, p )in enumerate(zip(grid, pts)):
            g_name = f'{g[0].item()},{g[1].item()}'
            dst = (p**2 - g**2).sum()
            if g_name in rcd:
                if dst > rcd[g_name]['dst']:
                    rcd[g_name]['dst'] = dst
                    rcd[g_name]['id'] = i
            else:
                rcd[g_name]={'dst': dst, 'id': i}
                
        
        grid_num = grid.shape[0]
        mask = im.new_zeros(grid_num).to(bool)
        valid_idx = torch.tensor([rcd[k]['id'] for k in rcd])
        mask[(valid_idx)] = True

    feat = []
    for g in grid[mask]:
        feat.append(im[:, g[1], g[0]]) # Note: the coordinate is (y, x)
    feat = torch.stack(feat, dim=0) # (n, c)

    return feat, mask


def fea_align(im, pts_norm, device=None):
    """
    :param im: feature map [c, w, h]
    :param pts: :[n, 2]
    :param device: output device. If not specified, it will be the same as the input
    :return: :interpolated feature vector [n,c]
    """
    if device is None:
        device = im.device
    pts_norm = pts_norm.to(torch.float32).to(device)
    C, H, W = im.shape
    # scale
    pts = pts_norm * torch.tensor([H, W]).to(pts_norm) 
    
    pts0 = torch.floor(pts)
    pts1 = pts0 + 1

    x, y = pts[:,0], pts[:,1]

    x0, y0 = pts0[:,0], pts0[:,1]
    x1, y1 = pts1[:,0], pts1[:,1]

    x0 = torch.clamp(x0, 0, im.shape[2] - 1)
    x1 = torch.clamp(x1, 0, im.shape[2] - 1)
    y0 = torch.clamp(y0, 0, im.shape[1] - 1)
    y1 = torch.clamp(y1, 0, im.shape[1] - 1)

    x0 = x0.to(torch.int32).to(device)
    x1 = x1.to(torch.int32).to(device)
    y0 = y0.to(torch.int32).to(device)
    y1 = y1.to(torch.int32).to(device)

    # Ia_idx = (y0, x0) # Note: the coordinate of the points are align to image, (y0,x0)=(0,0) on the left-up.
    # Ib_idx = (y1, x0)
    # Ic_idx = (y0, x1)
    # Id_idx = (y0, x1)

    Ia = get_fea_from_xy(im, y0, x0)
    Ib = get_fea_from_xy(im, y1, x0)
    Ic = get_fea_from_xy(im, y0, x1)
    Id = get_fea_from_xy(im, y0, x1)


    # # to perform nearest neighbor interpolation if out of bounds
    # if x0 == x1:
    #     if x0 == 0:
    #         x0 -= 1
    #     else:
    #         x1 += 1
    # if y0 == y1:
    #     if y0 == 0:
    #         y0 -= 1
    #     else:
    #         y1 += 1

    x0 = x0.to(torch.float32).to(device)
    x1 = x1.to(torch.float32).to(device)
    y0 = y0.to(torch.float32).to(device)
    y1 = y1.to(torch.float32).to(device)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    out = Ia * wa.unsqueeze(-1) + Ib * wb.unsqueeze(-1) + Ic * wc.unsqueeze(-1) + Id * wd.unsqueeze(-1)
    return out


def get_fea_from_xy(im, xs, ys):
    fea = []
    for x, y in zip(xs, ys):
        fea.append(im[:, x, y])
    fea = torch.stack(fea)
    return fea