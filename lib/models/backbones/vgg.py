import torch.nn as nn
from lib.build.registry import Registries

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]



class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm,  **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

@Registries.backbone_registry.register
def vgg11( **kwargs):
    """VGG 11-layer model (configuration "A")

    """
    return _vgg('vgg11', 'A', False,  **kwargs)

@Registries.backbone_registry.register
def vgg11_bn( **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    """
    return _vgg('vgg11_bn', 'A', True,  **kwargs)

@Registries.backbone_registry.register
def vgg13( **kwargs):
    """VGG 13-layer model (configuration "B")
    """
    return _vgg('vgg13', 'B', False,  **kwargs)

@Registries.backbone_registry.register
def vgg13_bn( **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    """
    return _vgg('vgg13_bn', 'B', True,  **kwargs)

@Registries.backbone_registry.register
def vgg16( **kwargs):
    """VGG 16-layer model (configuration "D")
    """
    return _vgg('vgg16', 'D', False,  **kwargs)

@Registries.backbone_registry.register
def vgg16_bn( **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    """
    return _vgg('vgg16_bn', 'D', True,  **kwargs)

@Registries.backbone_registry.register
def vgg19( **kwargs):
    """VGG 19-layer model (configuration "E")
    """
    return _vgg('vgg19', 'E', False,  **kwargs)

@Registries.backbone_registry.register
def vgg19_bn( **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    """
    return _vgg('vgg19_bn', 'E', True,  **kwargs)
