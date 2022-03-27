from utils import *
import torch
from train import corpus_bleu

BATCH = 20
max_length = 60
device = torch.device('cuda')


def get_ground_truth():
    results = []
    for i, data in enumerate(test_loader):
        (imgs, caption, caplen, all_captions) = data
        all_captions = [all_captions[i * CAP_PER_IMG].tolist() for i in range(BATCH)]

        for img_caps in all_captions:
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            results.append(img_captions)

        if i >= 10:
            break
    return results


rrrr = []
for model_file in MODEL_FILE_LIST:
    task_name = model_file.split('.')[0]
    encoder, decoder, test_loader, _, word_map = load_dataset_model(model_file, batch_size=BATCH * CAP_PER_IMG)

    ground_truth = get_ground_truth()

    for attack_name in ['L2', 'Linf']:
        adv_file = 'adv/' + str(0) + '_' + attack_name + '_' + task_name + '.adv'
        res = torch.load(adv_file)
        ori_seqs, adv_seqs = [], []
        for data in res:
            ori_img = data[0][0]
            adv_img = data[1][0]

            ori_seq, _ = prediction_batch(ori_img, encoder, decoder, word_map, max_length, device)
            for ori_s in ori_seq:
                ori_s = ori_s.tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        [ori_s]))  # remove <start> and pads
                ori_seqs.append(img_captions[0])

            adv_seq, _ = prediction_batch(adv_img, encoder, decoder, word_map, max_length, device)
            for adv_s in adv_seq:
                adv_s = adv_s.tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        [adv_s]))  # remove <start> and pads
                adv_seqs.append(img_captions[0])

        ori_bleu4 = corpus_bleu(ground_truth, ori_seqs)
        adv_bleu4 = corpus_bleu(ground_truth, adv_seqs)
        print(ori_bleu4, adv_bleu4)
        rrrr.append(np.array([ori_bleu4, adv_bleu4]).reshape([1, -1]))
rrrr = np.concatenate(rrrr, axis=0)
np.savetxt('acc.csv', rrrr, delimiter=',')




