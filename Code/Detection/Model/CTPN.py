from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from icecream import ic


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
        x = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        x = x.reshape((x.shape[0], x.shape[1], height, -1))
        return x


class CTPN(nn.Module):
    def __init__(self):
        super(CTPN, self).__init__()
        base_model = models.vgg16(pretrained=False)
        self.vgg = nn.Sequential(*(list(base_model.features)[:-1]))
        self.v2b = V2B(3, 1, 1)
        self.rnn = BiLSTM(3 * 3 * 512, 128)
        self.fc = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            nn.ReLU(inplace=True)
        )


        self.cls_score = nn.Conv2d(512, 2 * 10, 1)
        self.coordinate = nn.Conv2d(512, 2 * 10, 1)

    def forward(self, x):
        x = self.vgg(x)
        x = self.v2b(x)
        x = self.rnn(x)
        x = self.fc(x)

        cls_score = self.cls_score(x)
        cls_score  = cls_score.view(cls_score.size(0), cls_score.size(2)*cls_score.size(3)*10, 2)
        vertical_pred = self.coordinate(x)
        vertical_pred = vertical_pred.view(vertical_pred.size(0), vertical_pred.size(2)*vertical_pred.size(3)*10, 2)

        return cls_score, vertical_pred





if __name__ == '__main__':
    import torch
    t = torch.rand(1, 3, 700, 700)
    ctpn = CTPN()
    output = ctpn(t)
    ic(output[0].shape, output[1].shape)



