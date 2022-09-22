from torch import nn
import numpy as np
import torch.nn.functional as F
from icecream import ic

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        small_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        channels = 3

        layers = []
        for v in small_cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue
            else:
                layers += [nn.Conv2d(channels, v, 3, padding=1)]
                channels = v

            layers += [nn.ReLU(inplace=True)]

        self.feature = nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature(x)
        return out


class BiLSTM(nn.Module):
    def __init__(self, input_channel, hidden_size, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_channel, hidden_size, bidirectional=bidirectional)

    def forward(self, x):
        x = x.transpose(1, 3)
        recurrent, _ = self.lstm(x[0])  # therefor batch size should be 1
        recurrent = recurrent.unsqueeze(0)
        recurrent = recurrent.transpose(1, 3)
        return recurrent


class V2B(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(V2B, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


    def forward(self, x):
        height = x.shape[2]
        ic(x.shape)
        x = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        ic(x.shape)
        x = x.reshape((x.shape[0], x.shape[1], height, -1))
        ic(x.shape)
        return x


class CTPN(nn.Module):
    def __init__(self):
        super(CTPN, self).__init__()
        self.vgg = VGG()
        self.v2b = V2B(3, 1, 1)
        self.rnn = BiLSTM(3 * 3 * 512, 128)
        self.fc = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            nn.ReLU(inplace=True)
        )


        self.coordinate = nn.Conv2d(512, 2 * 10, 1)
        self.score = nn.Conv2d(512, 2 * 10, 1)
        self.refinement = nn.Conv2d(512, 10, 1)

    def forward(self, x):
        x = self.vgg(x)
        x = self.v2b(x)
        x = self.rnn(x)
        x = self.fc(x)

        vertical_pred = self.coordinate(x)
        score = self.score(x)
        refinement = self.refinement(x)

        return vertical_pred, score, refinement





if __name__ == '__main__':
    import torch
    t = torch.rand(1, 3, 128, 128)
    ctpn = CTPN()
    output = ctpn(t)
    ic(output[0].shape, output[1].shape)



