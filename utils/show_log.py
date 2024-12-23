import argparse


import argparse
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
import os
import os.path as osp

# matplotlib.use('TKAgg')

def show_loss(log_path, param, file_type='log', keyword=None, vis=True):
    if keyword is None:
        keyword = param
        file_name = param
    else:
        file_name = param+ ' of '+keyword
    for log in os.listdir(log_path):
        file_split = log.split('.')
        if not file_split[-1] == file_type:
            continue
        with open(osp.join(log_path, log)) as fp:
            data_list = []
            for line in fp.readlines():
                if keyword in line and param in line:
                    idx = line.find(param)+len(param)
                    try:
                        number = re.findall('\d+\.\d+|\d+', line[idx:idx+10])[0] #get the number
                        data_list.append(float(number))
                    except:
                        pass
        if len(data_list) == 0:
            continue
        y_data = np.array(data_list)
        save_path = osp.join(log_path, f'{file_name}_{file_split[0]}')
        max_x, min_x = np.max(y_data), np.min(y_data)
        max_idx, min_idx = np.argmax(y_data), np.argmin(y_data)
        
        if vis==True:
            x = np.arange(0, len(data_list)*2, 2)
            plt.plot(x, data_list)
            plt.annotate(f'{max_x}', (max_idx*2, max_x))
            plt.scatter(max_idx*2, max_x, c='red')
            plt.annotate(f'{min_x}', (min_idx*2, min_x))
            plt.scatter(min_idx*2, min_x, c='black')
            plt.xlabel("epoch")
            plt.title(file_name)
            plt.savefig(save_path)
            plt.close()
        print(f'The Param[{param}] of Key[{keyword}] is {max_x} in Epoch[{max_idx * 2}] ')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot the loss of results')
    parser.add_argument('path', type=str, help='relative path')
    parser.add_argument('--param', '-p', type=str, default='FC: top-1')
    parser.add_argument('--keyword', '-k', type=str, default='Retrieval')
    parser.add_argument('--type', type=str, default='log')
    # log_path = Path(os.getcwd()) /  'outputs/aux_adv/base_aux_adv_batch24'
    args = parser.parse_args()
    log_path = args.path
    # param = 'FC: top-1:' #'LOSS' #'FC: top-1:' 
    file_type = args.type
    param = args.param
    keyward = args.keyword
    
    show_loss(log_path, param, file_type, keyword=keyward)
