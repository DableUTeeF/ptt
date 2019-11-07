import os
import warnings
warnings.simplefilter("ignore")
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import MultiStepLR
from easydict import EasyDict
from utils import Progbar
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import f1_score
import numpy as np


def getmodel(cls):
    model = EfficientNet.from_pretrained('efficientnet-b6')
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, cls)
    return model


if __name__ == '__main__':
    args = EasyDict({
        'batch_size': 2,
        'batch_mul': 8,
        'val_batch_size': 2,
        'cuda': True,
        'model': '',
        'train_plot': False,
        'epochs': 180,
        'try_no': 'baselinev3_B6',
        'imsize': 528,
        'imsize_l': 550,
        'traindir': '/media/palm/62C0955EC09538ED/ptt/train/',
        'valdir': '/media/palm/62C0955EC09538ED/ptt/val/',
        'workers': 8,
        'resume': False,
        'multi_gpu': False
    })
    best_acc = 0
    best_no = 0
    start_epoch = 1
    model = getmodel(17).cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4,
                                )
    scheduler = MultiStepLR(optimizer, [2, 5])
    criterion = nn.CrossEntropyLoss().cuda()
    zz = 0
    train_dataset = datasets.ImageFolder(
        args.traindir,
        transforms.Compose([
            transforms.Resize(args.imsize),
            transforms.CenterCrop(args.imsize),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.valdir, transforms.Compose([
            transforms.Resize(args.imsize),
            transforms.CenterCrop(args.imsize),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.val_batch_size,
        num_workers=args.workers,
        pin_memory=False)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).cuda()
    # cudnn.benchmark = args.batch_size > 1
    if args.resume:
        if args.resume is True:
            args['resume'] = f'./checkpoint/{args.try_no}best.t7'
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['acc']
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    def train(epoch):
        print(f'\033[{np.random.randint(31, 37)}m', end='')
        print('\nEpoch: %d/%d' % (epoch, args.epochs))
        model.train()
        optimizer.zero_grad()
        criterion.zero_grad()
        progbar = Progbar(len(trainloader))
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs)
            loss = criterion(outputs, targets) / args.batch_mul
            try:
                loss.backward()
            except RuntimeError:
                print(' - error', inputs.size(), targets.size())
                continue
            _, predicted = outputs.max(1)
            f1_total = f1_score(targets.float().cpu().numpy(), predicted.float().cpu().numpy(), average='macro')
            lfs = (batch_idx + 1) % args.batch_mul
            if lfs == 0:
                optimizer.step()
                optimizer.zero_grad()
            suffix = [('loss', loss.item() * args.batch_mul),
                      ('acc', predicted.eq(targets).sum().item() / targets.size(0)),
                      ('f1', f1_total)]
            progbar.update(batch_idx + 1, suffix)

        optimizer.step()
        optimizer.zero_grad()


    def test(epoch):
        global best_acc, best_no
        model.eval()
        test_loss = 0
        progbar = Progbar(len(val_loader))
        predicteds = np.zeros((len(val_loader.sampler)), dtype='float32')
        targets_ = np.zeros(len(val_loader.sampler), dtype='float32')
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                f1 = f1_score(targets.float().cpu().numpy(), predicted.float().cpu().numpy(), average='macro')
                predicteds[batch_idx*args.val_batch_size:(batch_idx+1)*args.val_batch_size] = predicted.float().cpu().numpy()
                targets_[batch_idx*args.val_batch_size:(batch_idx+1)*args.val_batch_size] = targets.float().cpu().numpy()
                suffix = [('loss', loss.item()),
                          ('acc', predicted.eq(targets).sum().item() / targets.size(0)),
                          ('f1', f1)]
                progbar.update(batch_idx + 1, suffix)

        f1_total = f1_score(targets_, predicteds, average='macro')
        # Save checkpoint.
        acc = 100. * f1_total / len(val_loader)
        state = {
            'optimizer': optimizer.state_dict(),
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if acc > best_acc:
            torch.save(state, f'./checkpoint/{args.try_no}best.t7')
            best_acc = acc
        torch.save(model.state_dict(), f'./checkpoint/{args.try_no}temp.t7')


    for epoch in range(start_epoch, start_epoch + args.epochs):
        scheduler.step()
        train(epoch)
        test(epoch)
        print(f'best: {best_acc}')
