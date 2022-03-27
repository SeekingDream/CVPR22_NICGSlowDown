import os

import numpy as np
import torch

from utils import MODEL_FILE_LIST, MAX_PER_DICT


perDir = 'perRes'
if not os.path.isdir(perDir):
    os.mkdir(perDir)
all_res = []
for attack_name in ['L2', 'Linf']:
    for model_file in MODEL_FILE_LIST:
        task_name = model_file.split('.')[0]
        avg_res = np.zeros([3, 7])
        max_res = np.zeros([3, 7])
        delta_list = []
        for attack_type in range(7):
            adv_file = 'adv/' + str(attack_type) + '_' + attack_name + '_' + task_name + '.adv'
            res = torch.load(adv_file)
            ori_img, adv_img = [], []
            for data in res:
                ori_img.extend(data[0][0])
                adv_img.extend(data[1][0])
            ori_img = torch.stack(ori_img)
            adv_img = torch.stack(adv_img)
            delta = adv_img - ori_img
            delta = delta.reshape([len(delta), -1])
            if attack_name == 'L2':
                delta = torch.norm(delta, p=2, dim=1).unsqueeze(1)
            else:
                delta = torch.norm(delta, p=np.inf, dim=1).unsqueeze(1)
            delta = delta.cpu().numpy().reshape([-1, 1])
            delta_list.append(delta)
        delta_list = np.concatenate(delta_list, axis=1)

        final_res = np.concatenate([delta_list[:, 1:], delta_list[:, 0:1]], axis=1)
        file_name = os.path.join(perDir, task_name + '_' + attack_name + '.csv')
        np.savetxt(file_name, final_res, delimiter=',')
        print(file_name, 'success')
        all_res.append((final_res < MAX_PER_DICT[attack_name]).mean(0).reshape([-1, 1]))
print()
all_res = np.concatenate(all_res, axis=1)
file_name = os.path.join(perDir, 'average.csv')
np.savetxt(file_name, all_res, delimiter=',')

