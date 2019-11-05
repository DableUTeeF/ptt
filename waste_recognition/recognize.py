from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
from torch import nn
import torch
from torchvision import transforms


class Model:
    def __init__(self, num_classes, weights=None, label=None, input_size=528):
        self.model = self.from_name(num_classes)
        self.label = label
        self.input_size = input_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.Resize(self.input_size),
                                             transforms.CenterCrop(self.input_size),
                                             transforms.ToTensor(),
                                             self.normalize,
                                             ])

        if weights is not None:
            if not isinstance(weights, str):
                raise ValueError('`weights` should be string of path to saved weights file')
            self.load_weights(weights)

    def load_weights(self, weights):
        if not isinstance(weights, str):
            raise ValueError('`weights` should be string of path to saved weights file')
        checkpoint = torch.load(weights)
        if 'net' in checkpoint:
            self.model.load_state_dict(checkpoint['net'])
        else:
            self.model.load_state_dict(checkpoint)

    @staticmethod
    def from_name(cls):
        model = EfficientNet.from_name('efficientnet-b6')
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, cls)
        return model

    def predict(self, x, trasnform_first=True, normalize_first=True):
        if isinstance(x, np.ndarray):
            x = Image.fromarray(x)
        if trasnform_first:
            x = self.transform(x)
        elif normalize_first:
            x = transforms.ToTensor()(x)
            x = self.normalize(x)
        else:
            x = transforms.ToTensor()(x)
        x = x.view(1, 3, x.size(1), x.size(2))
        y = self.model(x).detach().numpy()
        if self.label is not None:
            cls = np.argmax(y, axis=1)
            return self.label[cls[0]]
        return y

    def predict_from_path(self, path):
        return self.predict(self.open_image(path))

    @staticmethod
    def open_image(path):
        return Image.open(path)
