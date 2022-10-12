import torch.nn as nn
import torch
import torch.nn.functional as F

class CTPN_loss(nn.Module):
    def __init__(self, device):
        super(CTPN_loss, self).__init__()
        self.device = device
        self.neg_pos_ratio  = 3
        self.class_loss = nn.CrossEntropyLoss()
        self.sigma = 0.9
        self.coordinate_loss = nn.L1Loss()
        self.refinement_loss = nn.L1Loss()


    def forward(self, cls, target_cls, regr, target_regr):
        # classification loss
        y_true = target_cls[0][0]
        cls_keep = (y_true != -1).nonzero()[:, 0]
        cls_true = y_true[cls_keep].long()
        cls_pred = cls[0][cls_keep]
        loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1),
                          cls_true)  # original is sparse_softmax_cross_entropy_with_logits
        # loss = nn.BCEWithLogitsLoss()(cls_pred[:,0], cls_true.float())  # 18-12-8
        cls_loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)
        cls_loss = cls_loss.to(self.device)

        # coordinate regression loss
        try:
            t_cls = target_regr[0, :, 0]
            t_regr = target_regr[0, :, 1:3]
            # apply regression to positive sample
            regr_keep = (t_cls == 1).nonzero()[:, 0]
            regr_true = t_regr[regr_keep]
            regr_pred = regr[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff < 1.0 / self.sigma).float()
            cor_loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1 - less_one) * (diff - 0.5 / self.sigma)
            cor_loss = torch.sum(cor_loss, 1)
            cor_loss = torch.mean(cor_loss) if cor_loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            # print(input, target)
            cor_loss = torch.tensor(0.0)
        cor_loss = cor_loss.to(self.device)

        # refinement regression loss
        # calculate side refinement regression loss


        return cls_loss, cor_loss

