import warnings
warnings.simplefilter("ignore")
import torch
import torch.nn as nn
import torch.utils.data.distributed
from efficientnet_pytorch import EfficientNet
import time


def getmodel(cls):
    model = EfficientNet.from_name('efficientnet-b6')
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, cls)
    return model


if __name__ == '__main__':
    model = getmodel(17)
    dummy = torch.rand((1, 3, 528, 528))
    starttime = time.time()
    runtime = starttime
    for i in range(100):
        model(dummy)
        print((time.time()-runtime))
        runtime = time.time()
    print((time.time()-starttime)/100)

