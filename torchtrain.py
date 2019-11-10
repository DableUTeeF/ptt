import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
from efficientnet_pytorch import EfficientNet
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import time
from sklearn.metrics import f1_score
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from utils import Progbar
from new_crop import crop
torch.manual_seed(0)
np.random.seed(0)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
    return res


def add_crop(image):
    image = np.array(image)
    return Image.fromarray((crop(image)).astype('uint8'))


if __name__ == '__main__':
    root = '/home/root1/Desktop/ott/'
    batch_size = 8
    val_transform = transforms.Compose([add_crop,
                                        transforms.Resize(528),
                                        transforms.CenterCrop(528),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
    train_transform = transforms.Compose([add_crop,
                                          transforms.Resize(528),
                                          transforms.CenterCrop(528),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

    trainset = torchvision.datasets.ImageFolder(root+'train', transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8)

    testset = torchvision.datasets.ImageFolder(root+'val', transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8)
    model = EfficientNet.from_pretrained('efficientnet-b6')
    model._fc = nn.Linear(model._fc.in_features, 17)
    model = nn.DataParallel(model)
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), 1e-2,
                                momentum=0.9,
                                weight_decay=1e-5)

    lr_schudule = ExponentialLR(optimizer, 0.97)

    def train(train_loader, model, criterion, optimizer, epoch, ):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = Progbar(len(train_loader))

        # switch to train mode
        model.train()

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # if i >= 200:
            #     print()
            #     break
            # measure data loading time
            data_time.update(time.time() - end)

            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            # losses.update(loss.item(), images.size(0))
            # top1.update(acc1[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            suffix = [('loss', loss.item()), ('acc', acc1[0].cpu().numpy())]
            progress.update(i+1, suffix)


    def validate(val_loader, model, criterion, ):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = Progbar(len(val_loader))

        # switch to evaluate mode
        model.eval()
        predicted = np.array([], dtype='float32')
        targets = np.array([], dtype='float32')
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images = images.cuda()
                target = target.cuda()

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                predicted = np.append(predicted, np.argmax(output, axis=1))
                targets = np.append(targets, target)
                acc1 = accuracy(output, target, topk=(1,))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                suffix = [('loss', loss.item()), ('acc', acc1[0].cpu().numpy())]
                progress.update(i + 1, suffix)
        f1 = f1_score(targets, predicted, average='macro')
        print(f1)
        return top1.avg, f1
    save_folder = len(os.listdir('checkpoint'))
    os.mkdir(os.path.join('checkpoint', str(save_folder)))
    for i in range(40):
        print(f'\033[{np.random.randint(31, 37)}m', 'Epoch:', i+1)
        train(trainloader, model, criterion, optimizer, i+1)
        torch.save(model.state_dict(), f'checkpoint/temp.torch')
        acc, f1 = validate(testloader, model, criterion)
        dct = {'net': model.state_dict(),
               'opt': optimizer.state_dict(),
               'acc': acc,
               'f1': f1
               }
        torch.save(dct, f'checkpoint/{save_folder}/{i}_{f1:.4f}.torch')
        lr_schudule.step()
