import torch.nn as nn
import numpy as np

from .captionAPI import *


class BaseAttack:
    def __init__(self, encoder, decoder, word_map, attack_norm, device, config):
        self.encoder = encoder
        self.decoder = decoder
        self.word_map = word_map
        self.attack_norm = attack_norm
        self.device = device
        self.config = config
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        self.decoder = self.decoder.to(self.device)
        self.decoder.eval()

        self.softmax = nn.Softmax(dim=1)
        self.bce_loss = nn.BCELoss(reduction='none')
        self.mse_Loss = nn.MSELoss(reduction='none')
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.max_iter = config['max_iter']
        self.max_per = config['max_per']
        self.beams = config['beams']
        self.max_len = config['max_len']
        self.lr = config['lr']

    def get_trans_len(self, x):
        seqs_len = prediction_len_batch(x, self.encoder, self.decoder, self.word_map, self.max_len, self.device)
        return np.array(seqs_len)

    def run_attack(self, x):
        pass

    def compute_loss(self, x):
        pass
