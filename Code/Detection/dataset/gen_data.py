import numpy as np
from PIL import Image
import random
import os
import matplotlib.pyplot as plt

dataset_dir = '../../dataset/Hanzi/CASIA-HWDB_Test/Test'
char_list = os.listdir(dataset_dir)


def draw_gt_boxes(img, gt_boxes):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(img, cmap='gray')
    for box in gt_boxes:
        _p = (box[0], box[1])
        _w = box[2] - box[0]
        _h = box[3] - box[1]
        rect = plt.Rectangle(_p, _w, _h, color='blue')
        print(rect)
        ax.add_patch(rect)
    plt.show()

def gen_train_data(f_path=None, scale=(0.5, 1.5), angle=30):
    def str2img(_s=None):

        special_char_dict = {
            # eng
            '*': 'asterisk',
            '\\': 'backslash',
            '/': 'forward_slash',
            '.': 'full_stop',
            '>': 'greater_than',
            '<': 'less_than',
            '|': 'vertical_bar',
            '?': 'question_mark',

            ':': 'colon',
            '"': 'double_quote',

            '_': '-',

            # chs
            '！': '!',
            '，': ',',
        }

        _img_list = []

        if not _s:
            # random select
            _l = random.randint(1, 15)
            for i in range(_l):
                char_path = os.path.join(dataset_dir, random.choice(char_list))
                char_path = os.path.join(char_path, random.choice(os.listdir(char_path)))
                _img_list.append(Image.open(char_path))

        for _c in _s.upper():
            if _c in char_list:
                char_path = os.path.join(dataset_dir, _c)
            elif _c in special_char_dict.keys():
                char_path = os.path.join(dataset_dir, special_char_dict[_c])
            else:
                char_path = os.path.join(dataset_dir, 'question_mark')
            char_path = os.path.join(char_path, random.choice(os.listdir(char_path)))
            _img_list.append(Image.open(char_path))

        return _img_list

    def gen_sentence(_img_list):
        # cal pic len
        _w = 0
        _h = 0
        for img in _img_list:
            _w += img.size[0]
            if img.size[1] > _h:
                _h = img.size[1]
        # gen pic
        _blank = Image.new('L', (_w, _h), color=255)
        width_index = 0
        for img in _img_list:
            _blank.paste(img, (width_index, 0))
            width_index += img.size[0]

        return _blank

    def gen_paragraph(file_path=None):
        ws = []
        hs = []
        imgs = []
        bboxes = []
        if not file_path:
            lines = random.randint(1, 4)
            for _ in range(lines):
                _img = gen_sentence(str2img())
                imgs.append(_img)
                _w, _h = _img.size
                _bbox = np.asarray([0, 0, 0, _h, _w, _h, _w, 0])
                _bbox[1::2] = _bbox[1::2] + sum(hs)
                hs.append(_h)
                ws.append(_w)
                bboxes.append(_bbox)
        else:
            with open(file_path, 'r+', encoding='utf-8') as f:
                lines = f.read().split('\n')
                hs = []
                imgs = []
                bbox = []
                for i in lines:
                    _img = gen_sentence(str2img(i))
                    imgs.append(_img)
                    _w, _h = _img.size
                    _bbox = np.asarray([0, 0, 0, _h, _w, _h, _w, 0])
                    _bbox[1::2] = _bbox[1::2] + sum(hs)
                    hs.append(_h)
                    ws.append(_w)
                    bboxes.append(_bbox)

        _blank = Image.new('L', (max(ws), sum(hs)), color=255)
        height_index = 0
        for img in imgs:
            _blank.paste(img, (0, height_index))
            height_index += img.size[1]
        return _blank, np.asarray(bboxes)

    _img, bbox = gen_paragraph(f_path)
    print(bbox)
    draw_gt_boxes(_img, bbox)

    # scale
    w, h = _img.size
    scale_fac = (random.random() / (scale[1] - scale[0]) + scale[0])  # (0.5x ~ 1.5x)
    _img = _img.resize((int(w * scale_fac), int(h * scale_fac)))

    # rotate
    angle = random.randint(-angle, angle)
    _img = _img.rotate(angle, expand=True, fillcolor=255)
    # add padding

    # noises

    return _img, bbox


if __name__ == '__main__':
    image, bbox = gen_train_data('test.txt')

