import datetime
import os
import time

import torch
import argparse


from utils import *

DEVICE_LIST = ['cuda', 'cpu']

ITER_DICT = {
    'cuda': 5,
    'cpu': 1
}


def test_efficiency(imgs, encoder, decoder, word_map, max_length):
    res = []
    for i, img in tqdm(enumerate(imgs)):
        if i > 100:
            break
        img = img.unsqueeze(0)
        device_res = []
        pred_len = 0
        for device in DEVICE_LIST:
            encoder = encoder.to(device).eval()
            decoder = decoder.to(device).eval()
            img = img.to(device)
            max_iter = ITER_DICT[device]
            t1 = time.time()
            for _ in range(max_iter):
                pred_len = prediction_len_batch(img, encoder, decoder, word_map, max_length, device)
            t2 = time.time()
            device_res.append(t2 - t1)
        device_res.append(pred_len[0])
        res.append(device_res)
    return res


def main(task_id, attack_type):

    if not os.path.isdir('latency'):
        os.mkdir('latency')


    model_file = MODEL_FILE_LIST[task_id]
    task_name = model_file.split('.')[0]

    encoder, decoder, _, _, word_map = load_dataset_model(model_file, batch_size=1 * CAP_PER_IMG)
    print('load model %s successful' % MODEL_FILE_LIST[task_id])
    for attack_name in ['L2', 'Linf']:
        adv_file = 'adv/' + str(attack_type) + '_' + attack_name + '_' + task_name + '.adv'
        res = torch.load(adv_file)
        ori_img, adv_img, ori_len, adv_len = [], [], [], []
        for data in res:
            ori_img.extend(data[0][0])
            adv_img.extend(data[1][0])
            ori_len.extend(data[0][1])
            adv_len.extend(data[1][1])
        ori_img = torch.stack(ori_img)
        adv_img = torch.stack(adv_img)
        ori_efficiency = test_efficiency(ori_img, encoder, decoder, word_map, max_length=60)
        adv_efficiency = test_efficiency(adv_img, encoder, decoder, word_map, max_length=60)
        torch.save(
            [ori_efficiency, adv_efficiency],
            os.path.join('latency', str(attack_type) + '_' + attack_name + '_' + task_name + '.latency')
        )
        print(str(attack_type) + '_' + attack_name + '_' + task_name + '   success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--task', default=0, type=int, help='experiment subjects')
    parser.add_argument('--attack', default=0, type=int, help='attack method')
    args = parser.parse_args()
    main(args.task, args.attack)
