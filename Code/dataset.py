import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

import matplotlib.pyplot as plt

from parse_gt import parse_gt
from CTPN_utils import draw_gt_boxes



class ICPRDataset(Dataset):
    def __init__(self, args):
        """
        load ICPR dataset
        :param path: path of the dataset directory, contains image_train and txt_train directories
        """
        self.path = args.path
        self.data_path = os.path.join(self.path, 'image_train')
        self.label_path = os.path.join(self.path, 'txt_train')
        self.img_names = os.listdir(os.path.join(self.path, 'image_train'))

    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, index):
        try:
            img_name = self.img_names[index]
            i = img_name.split('.')[:-1]
            img_name = '.'.join(i)
            del i
            # img
            img_path = os.path.join(self.data_path, self.img_names[index])
            img = Image.open(img_path)

            # label
            label_path = os.path.join(self.label_path, f'{img_name}.txt')
            gtbox = parse_gt(label_path)
        except:
            print('fail to load image, use default image')
            return self[0]

        draw_gt_boxes(img, gtbox)
        trans = T.ToTensor()
        img = trans(img)


        print(gtbox)
        return img, gtbox




if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader
    from config import load_config

    args = load_config()

    img_path = os.path.join(args.path, 'image_train')

    dataset = ICPRDataset(args)
    dataset = DataLoader(dataset, batch_size=1, shuffle=True)
    data, label = next(iter(dataset))
    data = np.array(data)
    data = data.squeeze()
    data = data.transpose([1, 2, 0])
    plt.imshow(data)
    plt.show()




