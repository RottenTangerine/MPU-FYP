import os
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as T

class ICPRDataset(Dataset):
    def __init__(self, path):
        """
        load ICPR dataset
        :param path: path of the dataset directory, contains image_train and txt_train directories
        """
        self.path = path
        self.data_path = os.path.join(path, 'image_train')
        self.label_path = os.path.join(path, 'txt_train')
        self.img_names = os.listdir(os.path.join(path, 'image_train'))

    def __len__(self):
        return len(self.img_names)


    def parse_label(self, path):
        label_lists = []
        with open(path, encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                label_list = line.split(',')[:8]
                label_lists.append(label_list)
        return label_lists


    def __getitem__(self, index):
        img_name = self.img_names[index]
        print(img_name)
        i = img_name.split('.')[:-1]
        img_name = '.'.join(i)
        # img
        img_path = os.path.join(self.data_path, self.img_names[index])
        img = Image.open(img_path)
        # h, w = img.size
        # rescale_fac = max(img.size) / 1600
        # if rescale_fac > 1.0:
        #     trans = T.Compose([T.Resize((int(h / rescale_fac), int(w / rescale_fac))),
        #                        T.ToTensor()])
        trans = T.ToTensor()
        img = trans(img)

        # label
        label_path = os.path.join(self.label_path, f'{img_name}.txt')
        gtbox = self.parse_label(label_path)

        # clip image:



        return img, gtbox




if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    dataset = ICPRDataset('Dataset/ICPR2018')
    dataset = DataLoader(dataset, batch_size=1)
    data, label = next(iter(dataset))
    data = np.array(data)
    data = data.squeeze()
    data = data.transpose([1, 2, 0])
    plt.imshow(data)
    plt.show()

    print(label)

