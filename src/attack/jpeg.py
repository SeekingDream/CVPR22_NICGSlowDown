import torch
import numpy as np

from .utils import *
from .baseattack import BaseAttack
from tqdm import tqdm
from .compress import jpeg_compression


class JPEGAttack(BaseAttack):
    def __init__(self, encoder, decoder, word_map, attack_norm, device, config):
        super(JPEGAttack, self).__init__(encoder, decoder, word_map, attack_norm, device, config)
        self.mean, self.std = 0, 1

    def run_attack(self, x):
        ori_img, ori_len = x.clone(), self.get_trans_len(x)
        adv_img = jpeg_compression(ori_img)
        adv_img = adv_img.to(self.device)
        adv_len = self.get_trans_len(adv_img)
        log_msg = 'inc: %.2f' % (sum(adv_len) / sum(ori_len))
        print(log_msg)
        return True, [ori_img, ori_len], [adv_img, adv_len]

