from config import load_config

from torch.utils.data import DataLoader, random_split
from dataset.dataset import ICPRDataset
from model.CTPN import CTPN
from model.loss import CTPN_loss

import torch
import time
import os

args = load_config()
train_id = int(time.time())
resume_epoch = 0

# dataset
dataset = ICPRDataset(args)
train_dataset, validate_dataset = random_split(dataset,
                                               [l:=round(len(dataset) * (1 - args.test_split_ratio)), len(dataset) - l])
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
validate_loader = DataLoader(dataset=validate_dataset, batch_size=args.batch_size, shuffle=True)

print(f'Train data: {len(train_dataset)}, Validate Data:{len(validate_dataset)}')

# initialize
device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
print(f'Current device: {device}')

model = CTPN().to(device)

criterion = CTPN_loss(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)


# continue training
try:
    most_recent_check_point = os.listdir('checkpoint')[-1]
    ckpt_path = os.path.join('checkpoint', most_recent_check_point)
    check_point = torch.load(ckpt_path)
    model.load_state_dict(check_point['state_dict'])
    resume_epoch = check_point['epoch']
    print(f'Successfully load checkpoint {most_recent_check_point}, '
          f'start training from epoch {resume_epoch + 1}, '
          f'loss: {check_point["loss"]}')
except:
    print('fail to load checkpoint, train from zero beginning')

# train
loss = 0
for epoch in range(resume_epoch + 1, args.epochs):
    for i, (imgs, classes, regrs) in enumerate(train_loader):
        imgs = imgs.to(device)
        classes = classes.to(device)
        regrs = regrs.to(device)

        out_cls, out_regr = model(imgs)

        optimizer.zero_grad()
        cls_loss, reg_loss = criterion(out_cls, classes, out_regr, regrs)
        loss = cls_loss + reg_loss
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Training Epoch: {epoch}/{args.epochs} [{i}/{len(train_loader)}]\t'
                  f'Loss: {loss.item():0.4f}\tLR: {optimizer.state_dict()["param_groups"][0]["lr"]:0.6f}')
    schedular.step()

    os.makedirs('checkpoint', exist_ok=True)
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'loss': loss,
                'lr': optimizer.state_dict()["param_groups"][0]["lr"],
                }, f'checkpoint/{train_id}_{epoch:03d}.pt')

# save model
os.makedirs('trained_model', exist_ok=True)
torch.save(model.state_dict(), f'./trained_model/CTPN_{train_id}.pth')
