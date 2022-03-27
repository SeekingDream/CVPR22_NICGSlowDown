import torch
import os
import numpy as np

from utils import *

l2_res = np.zeros([6, 7])
linf_res = np.zeros([6, 7])

resDir = 'res'
if not os.path.isdir(resDir):
    os.mkdir(resDir)
for model_file in MODEL_FILE_LIST:
    task_name = model_file.split('.')[0]
    for attack_name in ['L2', 'Linf']:
        avg_res = np.zeros([3, 7])
        max_res = np.zeros([3, 7])
        for attack_type in [1,2,3,4,5,6,0]:
            latency_file = os.path.join('latency', str(attack_type) + '_' + attack_name + '_' + task_name + '.latency')
            latency_res = torch.load(latency_file)
            ori_res, adv_res = latency_res
            cpu_inc, gpu_inc, loop_inc = [], [], []
            for ori, adv in zip(ori_res, adv_res):
                cpu_inc.append(adv[1] / ori[1] - 1)
                gpu_inc.append(adv[0] / ori[0] - 1)
                loop_inc.append(adv[2] / ori[2] - 1)
            cpu_inc, gpu_inc, loop_inc = np.array(cpu_inc), np.array(gpu_inc), np.array(loop_inc)

            avg_res[0, attack_type] = loop_inc.mean()
            avg_res[1, attack_type] = cpu_inc.mean()
            avg_res[2, attack_type] = gpu_inc.mean()

            max_res[0, attack_type] = loop_inc.max()
            max_res[1, attack_type] = cpu_inc.max()
            max_res[2, attack_type] = gpu_inc.max()
        final_res = np.concatenate([avg_res, max_res], axis=0)
        file_name = os.path.join(resDir, task_name + '_' + attack_name + '.csv')
        final_res = np.concatenate([final_res[:,1:], final_res[:, 0:1]], axis=1)
        np.savetxt(file_name, final_res, delimiter=',')
        print(file_name, 'success')
