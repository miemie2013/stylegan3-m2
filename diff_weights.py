

import torch
import numpy as np


'''
网络层中类似
self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
self.register_buffer('freqs', freqs)
登记的参数都不会被训练更新（除非强制赋值copy_()），但是会出现在权重state_dict里。
'''


model_00_dic_pytorch = torch.load("G_00.pth", map_location="cpu")
model_19_dic_pytorch = torch.load("G_19.pth", map_location="cpu")

for key, value in model_00_dic_pytorch.items():
    if 'synthesis.input' in key:
        v1 = model_00_dic_pytorch[key].numpy()
        v2 = model_19_dic_pytorch[key].numpy()
        ddd = np.sum((v1 - v2) ** 2)
        print('diff=%.6f (%s)' % (ddd, key))


