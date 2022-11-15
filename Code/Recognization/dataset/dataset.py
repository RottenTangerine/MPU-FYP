import os
import random

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

from Recognization.dataset.parse_gt import parse_gt
from Detection.dataset.CTPN_utils import draw_gt_boxes, gen_basic_anchor, cal_rpn


class ICPRDataset(Dataset):
    def __init__(self, args):
        """
        load ICPR dataset
        :param args: args contains path of the dataset directory, contains image_train and txt_train directories
        """
        self.args = args
        self.path = args.path
        self.data_path = os.path.join(self.path, 'image_train')
        self.label_path = os.path.join(self.path, 'txt_train')
        self.img_names = os.listdir(os.path.join(self.path, 'image_train'))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        i = img_name.split('.')[:-1]
        img_name = '.'.join(i)
        del i

        # read img
        img_path = os.path.join(self.data_path, self.img_names[index])
        img = cv2.imread(img_path)

        if img is not None:
            # rescale img
            h, w, c = img.shape
            rescale_fac = max(h, w) / 1600
            if rescale_fac > 1.0:
                h = int(h / rescale_fac)
                w = int(w / rescale_fac)
                img = cv2.resize(img, (w, h))

            # label
            label_path = os.path.join(self.label_path, f'{img_name}.txt')
            result = parse_gt(label_path, rescale_fac)

            images = []
            labels = []
            for *coor, label in result:
                sub_img = img[coor[1]:coor[3], coor[0]:coor[2]]
                images.append(sub_img)
                labels.append(label)

            image, label = random.choice(list(zip(images, labels)))
            image = torch.from_numpy(image.transpose([2, 0, 1])).float()

            return image, label
        else:
            # print(f'fail to load image: {self.img_names[index]}, use the default image')
            return self[0]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from Code.Detection.config import load_config
    import matplotlib.pyplot as plt
    from icecream import ic

    args = load_config()
    args.path = '../..//Dataset/ICPR2018'

    dataset = ICPRDataset(args)
    dataset = DataLoader(dataset, batch_size=1, shuffle=True)
    imgs, labels = next(iter(dataset))
    ic(imgs[0].shape, labels[0])
    plt.imshow(imgs[0])
    plt.show()
    ic(labels[0])
