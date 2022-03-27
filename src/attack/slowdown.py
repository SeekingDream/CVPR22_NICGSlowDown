from tqdm import tqdm
import torch.nn as nn
from torch import optim
import numpy as np

from .baseattack import BaseAttack
from .captionAPI import *
from .utils import L2, Linf


class SlowDownAttack(BaseAttack):
    def __init__(self, encoder, decoder, word_map, attack_norm, device, config):
        super(SlowDownAttack, self).__init__(encoder, decoder, word_map, attack_norm, device, config)
        self.coeff = config['coeff']

    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return torch.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    def compute_per(self, adv_imgs, ori_imgs):
        if self.attack_norm == L2:
            current_L = self.mse_Loss(
                self.flatten(adv_imgs),
                self.flatten(ori_imgs)).sum(dim=1)
        elif self.attack_norm == Linf:
            current_L = (self.flatten(adv_imgs) - self.flatten(ori_imgs)).max(1)[0]
        else:
            raise NotImplementedError
        return current_L

    def compute_adv_loss(self, adv_imgs):

        # seqs, seq_scores = prediction_batch_target_end(
        #     adv_imgs, self.encoder, self.decoder,
        #     self.word_map, self.max_len, self.device
        # )
        seqs, seq_scores = prediction_batch_target_end(
            adv_imgs, self.encoder, self.decoder,
            self.word_map, self.max_len, self.device
        )
        loss = self.bce_loss(seq_scores, torch.zeros_like(seq_scores))
        return loss.mean(1).sum()

    def run_attack(self, x):
        dim = len(x.shape)
        prev_cost = 1e10

        ori_img, ori_len = x.clone(), self.get_trans_len(x)
        best_adv, best_len = x.clone(), ori_len

        w = self.inverse_tanh_space(x).detach()
        w.requires_grad = True
        optimizer = optim.Adam([w], lr=self.lr)

        pbar = tqdm(range(self.max_iter))
        for it in pbar:
            adv_images = self.tanh_space(w)
            # Calculate loss
            adv_loss = self.compute_adv_loss(adv_images)
            current_per = self.compute_per(adv_images, ori_img)
            per_loss = self.relu(current_per - self.max_per)
            per_loss = per_loss.sum()

            loss = adv_loss + self.coeff * per_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update adversarial images
            current_len = self.get_trans_len(adv_images)

            is_best_adv = (current_len < best_len)
            mask = torch.tensor((1 - is_best_adv)).to(self.device) * (current_per.detach() < self.max_per)

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv = mask * adv_images.detach() + (1 - mask) * best_adv
            mask = mask.reshape([-1]).detach().cpu().numpy()
            best_len = mask * current_len + (1 - mask) * best_len

            # if it % (self.max_iter//10) == 0:
            #     if loss.item() > prev_cost:
            #         return False, best_adv
            #     prev_cost = loss.item()

            log_str = "i:%d,inc:%.2f,per:%.2f,adv_loss:%.2f,per_loss:%.2f" \
                      % (it, float(best_len.sum()) / float(ori_len.sum()), current_per.mean(), adv_loss, per_loss)
            pbar.set_description(log_str)
        return True, [ori_img, ori_len], [best_adv, best_len]

    def run_diff_loss(self, x, loss_type):
        dim = len(x.shape)
        ori_img, ori_len = x.clone(), self.get_trans_len(x)
        best_adv, best_len = x.clone(), ori_len

        w = self.inverse_tanh_space(x).detach()
        w.requires_grad = True
        optimizer = optim.Adam([w], lr=self.lr)

        pbar = tqdm(range(self.max_iter))
        for it in pbar:
            adv_images = self.tanh_space(w)
            adv_loss = self.compute_differ_loss(adv_images, loss_type)
            current_per = self.compute_per(adv_images, ori_img)
            per_loss = self.relu(current_per - self.max_per)
            per_loss = per_loss.sum()

            loss = adv_loss + self.coeff * per_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update adversarial images
            current_len = self.get_trans_len(adv_images)

            is_best_adv = (current_len < best_len)
            mask = torch.tensor((1 - is_best_adv)).to(self.device) * (current_per.detach() < self.max_per)

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv = mask * adv_images.detach() + (1 - mask) * best_adv
            mask = mask.reshape([-1]).detach().cpu().numpy()
            best_len = mask * current_len + (1 - mask) * best_len

            log_str = "i:%d,inc:%.2f,per:%.2f,adv_loss:%.2f,per_loss:%.2f" \
                      % (it, float(best_len.sum()) / float(ori_len.sum()), current_per.mean(), adv_loss, per_loss)
            pbar.set_description(log_str)
        return True, [ori_img, ori_len], [best_adv, best_len]

    def compute_differ_loss(self, adv_imgs, loss_type):
        if loss_type == 0:
            seqs, seq_scores = prediction_batch(
                adv_imgs, self.encoder, self.decoder,
                self.word_map, self.max_len, self.device
            )
        else:
            seqs, seq_scores = prediction_batch_end(
                adv_imgs, self.encoder, self.decoder,
                self.word_map, self.max_len, self.device
            )
        loss = self.bce_loss(seq_scores, torch.zeros_like(seq_scores))
        return loss.mean(1).sum()