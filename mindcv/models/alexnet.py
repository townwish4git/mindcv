"""
MindSpore implementation of `AlexNet`.
Refer to ImageNet Classification with Deep Convolutional Neural Networks
"""

from collections import OrderedDict

from mindspore import nn, Tensor

from .helpers import load_pretrained
from .registry import register_model


__all__ = [
    "AlexNet",
    "AlexNet4096",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 10,
        "first_conv": "features.0",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "AlexNet4096": _cfg(url=""),
}

class AlexNet(nn.Cell):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000
    ) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4, pad_mode="valid"), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, pad_mode="pad", padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, pad_mode="pad", padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, pad_mode="pad", padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, pad_mode="pad", padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell(
            nn.Dense(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dense(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dense(4096, num_classes)
        )
    
    def construct(self, x: Tensor):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
@register_model
def AlexNet4096(pretrained: bool = False, in_channels: int = 3, num_classes: int = 1000, **kwargs):
    default_cfg = default_cfgs["AlexNet4096"]
    model = AlexNet(in_channels=in_channels, num_classes=num_classes)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model