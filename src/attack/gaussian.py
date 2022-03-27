import torch
import numpy as np

from .utils import *
from .baseattack import BaseAttack
from tqdm import tqdm


class GaussianAttack(BaseAttack):
    def __init__(self, encoder, decoder, word_map, attack_norm, device, config):
        super(GaussianAttack, self).__init__(encoder, decoder, word_map, attack_norm, device, config)
        self.mean, self.std = 0, 1

    def run_attack(self, x):
        ori_img, ori_len = x.clone(), self.get_trans_len(x)

        ori_shape = ori_img.shape
        delta = torch.tensor(np.random.normal(size=ori_shape, scale=0.06))
        delta = delta.reshape([len(delta), -1])

        if self.attack_norm == L2:
            delta = delta / torch.norm(delta, p=2, dim=1).unsqueeze(1) * self.max_per
        else:
            delta = delta / torch.norm(delta, p=np.inf, dim=1).unsqueeze(1) * self.max_per

        delta = delta.reshape(ori_shape)
        delta = delta.to(self.device)

        adv_img = torch.clamp(ori_img + delta, min=0, max=1)
        adv_img = adv_img.to(torch.float32)
        adv_len = self.get_trans_len(adv_img)
        log_msg = 'inc: %.2f' % (sum(adv_len) / sum(ori_len))
        print(log_msg)

        return True, [ori_img, ori_len], [adv_img, adv_len]

