import torch
from tqdm import tqdm
import numpy as np

from .utils import *
from .captionAPI import *
from .baseattack import BaseAttack
from .compress import tvm

class TVMAttack(BaseAttack):
    def __init__(self, encoder, decoder, word_map, attack_norm, device, config):
        super(TVMAttack, self).__init__(encoder, decoder, word_map, attack_norm, device, config)

    def run_attack(self, imgs):
        ori_img, ori_len = imgs.clone(), self.get_trans_len(imgs)
        adv_imgs = []
        for img in imgs:
            adv_img = tvm(img.cpu())
            adv_imgs.append(adv_img)
        adv_imgs = torch.stack(adv_imgs)
        adv_imgs = adv_imgs.to(self.device)
        adv_len = self.get_trans_len(adv_imgs)
        log_msg = 'inc: %.2f' % (sum(adv_len) / sum(ori_len))
        print(log_msg)
        return True, [ori_img, ori_len], [adv_imgs, adv_len]