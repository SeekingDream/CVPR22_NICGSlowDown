import torch
import os
from torchvision import transforms

tensor2img = transforms.ToPILImage()


attack_type = 0
attack_name_list = ['L2', 'Linf']
task_name_list = [
    'BEST_coco_mobilenet_rnn',
    'BEST_coco_resnet_lstm',
    'BEST_flickr8k_googlenet_rnn',
    'BEST_flickr8k_resnext_lstm'
]
if not os.path.isdir('img'):
    os.mkdir('img')
if not os.path.isdir('img/delta'):
    os.mkdir('img/delta')
for attack_name in attack_name_list:
    for task_name in task_name_list:
        task_dir = 'img/' + attack_name + '_' + task_name
        if not os.path.isdir(task_dir):
            os.mkdir(task_dir)

        adv_file = 'adv/' + str(attack_type) + '_' + attack_name + '_' + task_name + '.adv'
        res = torch.load(adv_file)
        ori_img, adv_img, ori_len, adv_len = [], [], [], []
        for data in res:
            ori_img.extend(data[0][0])
            adv_img.extend(data[1][0])
            ori_len.extend(data[0][1])
            adv_len.extend(data[1][1])
        ori_imgs = torch.stack(ori_img)
        adv_imgs = torch.stack(adv_img)

        # for i, (ori_t, adv_t) in enumerate(zip(ori_imgs, adv_imgs)):
        #     ori_img = tensor2img(ori_t)
        #     adv_img = tensor2img(adv_t)
        #     ori_img.save(os.path.join(task_dir, str(i) + '_ori.jpg'))
        #     adv_img.save(os.path.join(task_dir, str(i) + '_adv.jpg'))
        #
        if 'coco' in task_dir:
            coco_id = [128, 130, 181, 61]
        else:
            coco_id = [55, 90]
        for coco in coco_id:
            delta = ori_imgs[coco] - adv_imgs[coco]
            delta_img = tensor2img(delta)
            delta_img.save(os.path.join('img/delta', attack_name + '_' + str(coco) + '_delta.jpg'))
        print(task_dir, 'successful')




