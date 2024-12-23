import os, json, logging
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import torch
from easydict import EasyDict as edict


def to_cuda(d):
    """
    recursively move tensor to cuda
    """
    if isinstance(d, list):
        return [to_cuda(v) for v in d]
    elif isinstance(d, tuple):
        return type(d)(to_cuda(s) for s in d)
    elif isinstance(d, dict):
        return {key: to_cuda(d[key]) for key in d}
    elif isinstance(d, torch.Tensor):
        return d.to('cuda')
    
def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, tuple):
        return type(obj)(move_to(v, device) for v in obj)
    else: 
        return obj
    # else:
    #     raise TypeError("Invalid type for move_to")
  
def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of opt image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label

def gen_idx_sort(train_opt_label):
    idx = [[x] for x in range(len(train_opt_label))]
    return idx, idx


def GenIdx(train_opt_label, train_sar_label):
    opt_pos = []
    unique_label_opt = np.unique(train_opt_label)
    for i in range(len(unique_label_opt)):
        tmp_pos = [k for k, v in enumerate(train_opt_label) if v == unique_label_opt[i]]
        opt_pos.append(tmp_pos)

    sar_pos = []
    unique_label_sar = np.unique(train_sar_label)
    for i in range(len(unique_label_sar)):
        tmp_pos = [k for k, v in enumerate(train_sar_label) if v == unique_label_sar[i]]
        sar_pos.append(tmp_pos)
    return opt_pos, sar_pos

def GenCamIdx(gall_img, gall_label, mode):
    if mode == 'indoor':
        camIdx = [1, 2]
    else:
        camIdx = [1, 2, 4, 5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))

    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k, v in enumerate(gall_label) if v == unique_label[i] and gall_cam[k] == camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos


def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
        # cam_id = 2
        gall_cam.append(cam_id)

    return np.array(gall_cam)


class MultiSampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_opt_label, train_sar_label: labels of two modalities
            opt_pos, sar_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_opt_label, train_sar_label, opt_pos, sar_pos, batchSize, per_img):
        uni_label = np.unique(train_opt_label)
        self.n_classes = len(uni_label)

        sample_opt = np.arange(batchSize)
        sample_sar = np.arange(batchSize)
        N = np.maximum(len(train_opt_label), len(train_sar_label))

        # per_img = 4
        per_id = batchSize / per_img
        for j in range(N // batchSize + 1):
            batch_idx = np.random.choice(uni_label, int(per_id), replace=False)
            for s, i in enumerate(range(0, batchSize, per_img)):
                #the train labels begin with 1, index should begin with 0
                sample_opt[i:i + per_img] = np.random.choice(opt_pos[batch_idx[s]-1], per_img, replace=False)
                sample_sar[i:i + per_img] = np.random.choice(sar_pos[batch_idx[s]-1], per_img, replace=False)
            if j == 0:
                index1 = sample_opt
                index2 = sample_sar
            else:
                index1 = np.hstack((index1, sample_opt))
                index2 = np.hstack((index2, sample_sar))
        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_opt_label, train_sar_label: labels of two modalities
            opt_pos, sar_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_opt_label, train_sar_label, batchSize, per_img):
        uni_label = np.unique(train_opt_label)
        self.n_classes = len(uni_label)
        N = np.maximum(len(train_opt_label), len(train_sar_label))
        # per_img = 4
        idx_list = []
        per_id = batchSize / per_img
        for j in range(N // batchSize + 1):
            batch_idx = np.random.choice(uni_label, int(per_id), replace=False)
            idx_list.append(batch_idx)
        index = np.hstack(idx_list)
        self.index = index
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index)))

    def __len__(self):
        return self.N

class SARSampler(Sampler):
    """Sample SAR identities evenly in each batch.
        Args:
            train_label, train_sar_label: labels of two modalities
            opt_pos, sar_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_sar_ids, batch_size, per_img, num_sample=None):
        sar_id = np.array(train_sar_ids)
        if num_sample is None:
            num_sample = len(sar_id) 
        sar_pos = np.arange(0, len(sar_id)) #build the pos index of all sar samples
        # per_img = 4
        idx_list = []
        per_id = batch_size / per_img

        while len(idx_list) <  (num_sample//batch_size + 1): #sample
            batch_pos = np.random.choice(sar_pos, int(per_id), replace=False)
            batch_id = sar_id[batch_pos]
            if batch_id.size == np.unique(batch_id).size: #check if the sar files are unique
                idx_list.append(batch_pos)

        index = np.hstack(idx_list)
        self.index = index
        self.N = num_sample
        self.n_classes = len(sar_id)

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return len(self.index)

class SortIDSampler(Sampler):
    """ sort the SAR id in abscending
        Args:
           train_sar_label: Id of SAR
    """

    def __init__(self, train_sar_ids):
        sar_id = np.array(train_sar_ids)
        index = np.argsort(sar_id)
        self.index = index

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return len(self.index)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            print('ERROR!')
            # if e.errno != errno.EEXIST:
            #     raise


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def update_args(args):
    r"""
    update the easydict args from argsparse and yaml based on wasy dict"""
    # default_cfg = edict()
    import yaml
    with open(args.cfg, 'r') as f:
        opt = edict(yaml.safe_load(f))

    for k, v in args.__dict__.items():
        opt[k] = v

    return opt


class Recorder:
    def __init__(self, record_dict) -> None:
        self.meter_dict = dict()
        self.param_dict = dict()
        self.vis_func = []
        for key, item in record_dict.items():
            self.meter_dict[key] = dict(type=item, meter=AverageMeter(), vis=True)

    def reset(self):
        for key, item in self.meter_dict.items():
            item['meter'].reset()

    def update(self, key, *args, type='f'):
        if key not in self.meter_dict:
            self.meter_dict[key] = dict(type=type, meter=AverageMeter(), vis=True)
        self.meter_dict[key]['meter'].update(*args)

    
    def display(self):
        rcd_list = []
        for key, item in self.meter_dict.items():
            if item['type'] == 'd':
                rcd_list.append(f"{key}: {int(item['meter'].avg):d}")
            elif item['type'] == 'f':
                rcd_list.append(f"{key}: {item['meter'].avg:.4f}")
            elif item['type'] == '%':
                rcd_list.append(f"{key}: {item['meter'].avg:.2%}")
        return rcd_list

    def save(self, **kwargs):
        self.param_dict.update(**kwargs)

    def get(self, key):
        return self.param_dict[key]


    def register_vis(self, func):
        self.vis_func.append(func)

    def param_vis(self, *args):
        for func in self.vis_func:
            func(self.param_dict, *args)

        
class Register:
    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()
    
MODEL = Register('model')
DATASET = Register('dataset')
OPTIM = Register('optimizer')
LOSS = Register('loss')

