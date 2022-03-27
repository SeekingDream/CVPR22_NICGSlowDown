from .compress import *
from .baseattack import BaseAttack


class Quantize(BaseAttack):
    def __init__(self, depth=8):
        super(Quantize, self).__init__()
        self.depth = depth

    def transform(self, image):
        return quantize_img(image, self.depth)


class Jpeg(BaseAttack):
    def __init__(self):
        super(Jpeg, self).__init__()

    def transform(self, image):
        return jpeg_compression(image)


class Tvm(BaseAttack):
    def __init__(self):
        super(Tvm, self).__init__()

    def transform(self, image):
        return tvm(image)


class TvmChambolle(BaseAttack):
    def __init__(self):
        super(TvmChambolle, self).__init__()

    def transform(self, image):
        return tvm_chambolle(image)


class TvmBregman(BaseAttack):
    def __init__(self):
        super(TvmBregman, self).__init__()

    def transform(self, image):
        return tvm_bregman(image)


class TvmL2(BaseAttack):
    def __init__(self):
        super(TvmL2, self).__init__()

    def transform(self, image):
        return tvm_tvl2(image)


class TvmLinf(BaseAttack):
    def __init__(self):
        super(TvmLinf, self).__init__()

    def transform(self, image):
        return tvm_tvinf(image)
