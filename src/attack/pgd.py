import torch
from tqdm import tqdm
import numpy as np

from .utils import *
from .captionAPI import *
from .baseattack import BaseAttack


class PGDAttack(BaseAttack):
    def __init__(self, encoder, decoder, word_map, attack_norm, device, config):
        super(PGDAttack, self).__init__(encoder, decoder, word_map, attack_norm, device, config)
        self.coeff = config['coeff']
        self.alpha = 0.01

    def compute_loss(self, adv_imgs):

        seqs, seq_scores = prediction_batch(
            adv_imgs, self.encoder, self.decoder,
            self.word_map, self.max_len, self.device
        )
        # seqs, seq_scores = prediction_batch_target_end(
        #     adv_imgs, self.encoder, self.decoder,
        #     self.word_map, self.max_len, self.device
        # )
        seq_len = [get_seq_len(seq, self.word_map) for seq in seqs]
        first_score = [seq_scores[i, 1] for i in range(len(seq_len))]
        first_score = torch.stack(first_score)
        loss = self.bce_loss(first_score, torch.zeros_like(first_score))
        return seq_len, loss.sum()

    def run_attack(self, x):
        ori_img, ori_len = x.clone(), self.get_trans_len(x)
        adv_img = x.clone()

        pbar = tqdm(range(self.max_iter))
        for it in pbar:
            adv_img.requires_grad = True
            current_len, loss = self.compute_loss(adv_img)
            grad = torch.autograd.grad(loss, adv_img,
                                       retain_graph=False, create_graph=False)[0]
            adv_img = adv_img.detach() + self.alpha * grad.sign()
            delta = adv_img - ori_img
            ori_shape = delta.shape
            delta = delta.reshape([len(delta), -1])
            if self.attack_norm == L2:
                delta = delta / torch.norm(delta, p=2, dim=1).unsqueeze(1) * self.max_per
            else:
                delta = delta / torch.norm(delta, p=np.inf, dim=1).unsqueeze(1) * self.max_per

            delta = delta.reshape(ori_shape)

            adv_img = torch.clamp(ori_img + delta, min=0, max=1).detach()
            log_msg = 'it: %d, inc: %.2f' % (it, sum(current_len) / sum(ori_len))
            pbar.set_description(log_msg)
        adv_len = self.get_trans_len(adv_img)
        return True, [ori_img, ori_len], [adv_img, adv_len]
