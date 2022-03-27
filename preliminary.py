import os.path

from utils import *

from torchvision import transforms

tensor2img = transforms.ToPILImage()

if not os.path.isdir('case'):
    os.mkdir('case')

def decode(index2word, seq):
    res = ''
    for s in seq:
        res += ' ' + index2word[int(s)]
    return res


def reverse(word_map):
    new_dict = {}
    for k in word_map:
        new_dict[word_map[k]] = k
    return new_dict


for model_file in TEST_FILE_LIST:
    _, _, test_loader, train_loader, word_map = load_dataset_model(model_file, 5, 1)
    index2word = reverse(word_map)
    m_rate, m_i, m_img = 0, None, None
    max_l, min_l = 0, 100000
    max_l_img, min_l_img = None, None
    max_l_text, min_l_text = None, None
    r_max_text, r_min_text = None, None
    for i, data in tqdm(enumerate(train_loader)):
        (imgs, cap, caplens) = data
        min_len = float(min(caplens))
        max_len = float(max(caplens))
        rate = max_len / min_len
        if rate > m_rate:
            m_rate = rate
            m_i = i
            m_img = imgs[0]
            r_max_text = decode(index2word, cap[caplens.argmax()])
            r_min_text = decode(index2word, cap[caplens.argmin()])

        if min_len < min_l:
            min_l = min_len
            min_l_img = imgs[0]
            min_l_text = decode(index2word, cap[caplens.argmin()])

        if max_len > max_l:
            max_l = max_len
            max_l_img = imgs[0]
            max_l_text = decode(index2word, cap[caplens.argmax()])

    max_l_img = tensor2img(max_l_img)
    min_l_img = tensor2img(min_l_img)
    m_img = tensor2img(m_img)

    max_l_img.save(os.path.join('case', model_file + 'max_l_text' + '.jpg'))
    print(model_file + max_l_text)
    min_l_img.save(os.path.join('case', model_file + 'min_l_text' + '.jpg'))
    print(model_file + min_l_text)
    m_img.save(os.path.join('case', model_file + 'r_max_text' + '.jpg'))
    print(model_file + r_max_text)
    m_img.save(os.path.join('case', model_file + 'r_min_text' + '.jpg'))
    print(model_file + r_min_text)
