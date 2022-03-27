import torch
import math
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from .tvm import reconstruct

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO


def quantize_img(im, depth=8):
    assert torch.is_tensor(im)
    N = int(math.pow(2, depth))
    im = (im * N).round()
    im = im / N
    return im


def jpeg_compression(imgs):
    assert torch.is_tensor(imgs)
    imgs = [ToPILImage()(i) for i in imgs]
    savepath = BytesIO()
    compress_im = []
    for im in imgs:
        im.save(savepath, 'JPEG', quality=75)
        im = Image.open(savepath)
        im = ToTensor()(im)
        compress_im.append(im)
    compress_im = torch.stack(compress_im, dim=0)
    return compress_im


def tvm(img, drop_rate=0.5, weight=0.03, lab=False, verbose=False):
    return reconstruct(img, drop_rate, 'none', weight, lab=lab, verbose=verbose)


def tvm_chambolle(img, drop_rate=0.5, weight=0.03, lab=False, verbose=False):
    return reconstruct(img, drop_rate, 'chambolle', weight, lab=lab, verbose=verbose)


def tvm_bregman(img, drop_rate=0.0, weight=0.03, lab=False, verbose=False):
    return reconstruct(img, drop_rate, 'bregman', weight, lab=lab, verbose=verbose)


def tvm_tvl2(img, drop_rate=0.5, weight=0.03, lab=False, verbose=False):
    return reconstruct(img, drop_rate, 'tvl2', weight, lab=lab, verbose=verbose)


def tvm_tvinf(img, drop_rate=0.5, weight=0.03, lab=False, verbose=False):
    return reconstruct(img, drop_rate, 'tvinf', weight, lab=lab, verbose=verbose)
