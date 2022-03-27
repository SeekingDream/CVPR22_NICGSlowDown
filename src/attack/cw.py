import torch
from tqdm import tqdm

from .baseattack import BaseAttack
from .captionAPI import prediction_batch, get_seq_len
from .utils import *


class CWAttack(BaseAttack):
    def __init__(self, encoder, decoder, word_map, attack_norm, device, config):
        super(CWAttack, self).__init__(encoder, decoder, word_map, attack_norm, device, config)
        self.coeff = config['coeff']

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return torch.atanh(x * 2 - 1)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

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
        seqs, seq_scores = prediction_batch(
            adv_imgs, self.encoder, self.decoder,
            self.word_map, self.max_len, self.device
        )
        seq_len = [get_seq_len(seq, self.word_map) for seq in seqs]

        first_score = [seq_scores[i, 1] for i in range(len(seq_len))]
        first_score = torch.stack(first_score)
        loss = self.bce_loss(first_score, torch.zeros_like(first_score))
        return seq_len, loss.sum()

    def run_attack(self, x):
        dim = len(x.shape)
        ori_img, ori_len = x.clone(), self.get_trans_len(x)
        best_adv, best_len = x.clone(), ori_len

        w = self.inverse_tanh_space(x).detach()
        w.requires_grad = True
        optimizer = torch.optim.Adam([w], lr=self.lr)

        pbar = tqdm(range(self.max_iter))
        for it in pbar:
            adv_images = self.tanh_space(w)
            # Calculate loss

            seq_len, adv_loss = self.compute_adv_loss(adv_images)
            current_per = self.compute_per(adv_images, ori_img)
            per_loss = current_per.sum()
            loss = adv_loss + self.coeff * per_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
