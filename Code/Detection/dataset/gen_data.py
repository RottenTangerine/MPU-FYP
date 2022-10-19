import numpy as np
from PIL import Image
import random
import os
import matplotlib.pyplot as plt
from icecream import ic

dataset_dir = '../../dataset/Hanzi/CASIA-HWDB_Test/Test'
char_list = os.listdir(dataset_dir)


def draw_gt_boxes(img, gt_boxes):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(img, cmap='gray')
    for box in gt_boxes:
        _p = (box[0], box[1])
        _w = box[4] - box[0]
        _h = box[3] - box[1]
        rect = plt.Rectangle(_p, _w, _h, color='blue', alpha=0.2)

        ax.add_patch(rect)
    plt.show()


def gen_train_data(f_path=None, scale_range=(0.5, 1.5), max_angle=30):
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

        else:
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
        for _img in _img_list:
            _w += _img.size[0]
            if _img.size[1] > _h:
                _h = _img.size[1]
        # gen pic
        _blank = Image.new('L', (_w, _h), color=255)
        width_index = 0
        for _img in _img_list:
            _blank.paste(_img, (width_index, 0))
            width_index += _img.size[0]

        return _blank

    def gen_paragraph(p=None):
        w_max = 0
        hs = []
        imgs = []
        bboxes = []
        if not p:
            for i in range(random.randint(1, 3)):
                _img = gen_sentence(str2img())

        else:
            for i in [p[i * 20: (i + 1) * 20] for i in range(len(p) // 20 + 1)]:
                _img = gen_sentence(str2img(i))
                imgs.append(_img)
                _w, _h = _img.size

                _bbox = np.asarray([0, 0, 0, _h, _w, _h, _w, 0])
                _bbox[1::2] = _bbox[1::2] + sum(hs)

                if _w > w_max:
                    w_max = _w
                hs.append(_h)
                bboxes.append(_bbox)

        _blank = Image.new('L', (w_max, sum(hs)), color=255)
        height_index = 0
        for _img in imgs:
            _blank.paste(_img, (0, height_index))
            height_index += _img.size[1]

        _img, _bbox = transform(_blank, np.asarray(bboxes))
        return _img, _bbox

    def gen_page(file_path=None, paragraph_blank=100):
        w_max = 0
        hs = []
        imgs = []
        bboxes = None
        if not file_path:
            for i in range(random.randint(1, 5)):
                _img, _bbox = gen_paragraph()
                imgs.append(_img)
                _bbox[:, 1::2] = _bbox[:, 1::2] + sum(hs)

                _w, _h = _img.size
                if _w > w_max:
                    w_max = _w
                hs.append(_h)
                hs.append(paragraph_blank)

                if not bboxes:
                    bboxes = _bbox
                else:
                    bboxes = np.vstack((bboxes, _bbox))
        else:
            with open(file_path, 'r+', encoding='utf-8') as f:
                lines = f.read().split('\n')
                for i in lines:
                    if not i:
                        continue
                    _img, _bbox = gen_paragraph(i)
                    _img, _bbox = transform(_img, _bbox, *transform_init())
                    imgs.append(_img)
                    _bbox[:, 1::2] = _bbox[:, 1::2] + sum(hs)

                    _w, _h = _img.size
                    if _w > w_max:
                        w_max = _w
                    hs.append(_h)
                    hs.append(paragraph_blank)

                    if bboxes is None:
                        bboxes = _bbox
                    else:
                        bboxes = np.vstack((bboxes, _bbox))

        _blank = Image.new('L', (w_max, sum(hs)), color=255)
        height_index = 0
        for _img in imgs:
            _blank.paste(_img, (0, height_index))
            height_index += _img.size[1] + paragraph_blank
        return _blank, np.asarray(bboxes)

    def transform(_img, _bbox, _scale=1, _angle=0, padding=0):

        # scale
        if _scale != 1:
            w, h = _img.size
            _img = _img.resize((int(w * _scale), int(h * _scale)))

            _bbox = _bbox * _scale

        # rotate
        if _angle != 0:
            w, h = _img.size
            cx, cy = w / 2, h / 2
            _img = _img.rotate(_angle, expand=True, fillcolor=255)
            _angle = np.radians(_angle)

            img_box = [0, 0, 0, h, w, h, w, 0]
            _bbox = np.vstack((img_box, _bbox))

            bbox_x = _bbox[:, ::2]
            bbox_y = _bbox[:, 1::2]
            bbox_x_new = (np.cos(_angle) * (bbox_x - cx)) + (np.sin(_angle) * (bbox_y - cy))
            bbox_y_new = (np.sin(_angle) * (bbox_x - cx)) - (np.cos(_angle) * (bbox_y - cy))
            bbox_x_new = bbox_x_new[1:] - min(bbox_x_new[0])
            bbox_y_new = _img.size[1] - (bbox_y_new[1:] - min(bbox_y_new[0]))
            x_min = np.min(bbox_x_new, axis=1).reshape(-1, 1)
            x_max = np.max(bbox_x_new, axis=1).reshape(-1, 1)
            y_min = np.min(bbox_y_new, axis=1).reshape(-1, 1)
            y_max = np.max(bbox_y_new, axis=1).reshape(-1, 1)

            _bbox = np.hstack((x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min))

        # add padding
        if padding != 0:
            w, h = _img.size
            _blank = Image.new('L', (w + 2 * padding, h + 2 * padding), color=255)
            _blank.paste(_img, (padding, padding))
            _img = _blank
            _bbox += padding

        # noises

        return _img, _bbox

    def transform_init():nn;.l;.
        scale = random.uniform(0.8, 1.2)
        angle = random.uniform(-10, 10)
        padding = random.randint(0, 150)

        return scale, angle, padding



    img, bbox = gen_page(f_path)
    # draw_gt_boxes(img, bbox)

    return img, bbox


if __name__ == '__main__':
    _image, gt_box = gen_train_data('test.txt')
    _image.save('../imgs/i_002.jpg')

