import torch
import os
import numpy as np

from utils import *

resDir = 'dist'
if not os.path.isdir(resDir):
    os.mkdir(resDir)
for model_file in MODEL_FILE_LIST:
    task_name = model_file.split('.')[0]
    for attack_name in ['L2', 'Linf']:
        avg_res = np.zeros([3, 7])
        max_res = np.zeros([3, 7])
        for attack_type in [0]:
            latency_file = os.path.join('latency', str(attack_type) + '_' + attack_name + '_' + task_name + '.latency')
            latency_res = torch.load(latency_file)
            ori_res, adv_res = latency_res

            res = []
            for ori, adv in zip(ori_res, adv_res):
                tmp = np.array(ori + adv).reshape([1, -1])
                res.append(tmp)

            res = np.concatenate(res, axis=0)

            file_name = os.path.join(resDir, '____dist____' + task_name + '_' + attack_name + '.csv')

            np.savetxt(file_name, res, delimiter=',')
            print(file_name, 'success')