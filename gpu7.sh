
# CUDA_VISIBLE_DEVICES=7 python train.py --config=flickr8k_resnext_lstm.json


#
#CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=0 --norm=0
#CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=0 --norm=1
#CUDA_VISIBLE_DEVICES=7 python test_latency.py --task=3 --attack=0

#CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=1 --norm=0
#CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=1 --norm=1

#CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=2 --norm=0
#CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=2 --norm=1
#
#CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=3 --norm=0
#CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=3 --norm=1
#
#CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=4 --norm=0
#CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=4 --norm=1
#
#CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=5 --norm=0
#CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=5 --norm=1
#
#CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=6 --norm=0
#CUDA_VISIBLE_DEVICES=7 python generate_adv.py --task=3 --attack=6 --norm=1
#

#CUDA_VISIBLE_DEVICES=7 python test_latency.py --task=3 --attack=1
#CUDA_VISIBLE_DEVICES=7 python test_latency.py --task=3 --attack=2
#CUDA_VISIBLE_DEVICES=7 python test_latency.py --task=3 --attack=3
#CUDA_VISIBLE_DEVICES=7 python test_latency.py --task=3 --attack=4
#CUDA_VISIBLE_DEVICES=7 python test_latency.py --task=3 --attack=5
#CUDA_VISIBLE_DEVICES=7 python test_latency.py --task=3 --attack=6
#

CUDA_VISIBLE_DEVICES=7 python loss_impact.py --task=3