import os

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

from Detection.dataset.parse_gt import parse_gt
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
            gtbox = parse_gt(label_path, rescale_fac)
            # draw_gt_boxes(img, gtbox)

            [cls, regr], base_anchors = cal_rpn(self.args, (h, w), (int(h / 16), int(w / 16)), 16, gtbox)

            m_img = img - self.args.IMAGE_MEAN

            regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

            cls = np.expand_dims(cls, axis=0)

            # transform to torch tensor
            m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
            cls = torch.from_numpy(cls).float()
            regr = torch.from_numpy(regr).float()

            return m_img, cls, regr

        else:
            # print(f'fail to load image: {self.img_names[index]}, use the default image')
            return self[0]


if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader
    from Code.Detection.config import load_config
    from icecream import ic

    args = load_config()
    args.path = '../..//Dataset/ICPR2018'

    dataset = ICPRDataset(args)
    dataset = DataLoader(dataset, batch_size=1, shuffle=True)
    imgs, classes, regrs = next(iter(dataset))
