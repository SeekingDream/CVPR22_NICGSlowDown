from .baseattack import BaseAttack
from tqdm import tqdm


class DeepFoolAttack(BaseAttack):
    def __init__(self, encoder, decoder, device, config):
        super(DeepFoolAttack, self).__init__(encoder, decoder, device, config)
