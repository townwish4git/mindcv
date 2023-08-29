"""
MindSpore implementation of `LeNet5`.
Refer to Gradient-Based Learning Applied to Document Recognition
"""

from collections import OrderedDict

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .helpers import load_pretrained
from .registry import register_model

__all__ = [
    "LeNet",
    "LeNet5",
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
    "LeNet5": _cfg(url=""),
}


class LeNet(nn.Cell):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10
    ) -> None:
        super(LeNet, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, pad_mode="pad", padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, pad_mode="pad", padding=0), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell([
            nn.Dense(400, 120), nn.ReLU(),
            nn.Dense(120, 84), nn.ReLU(),
            nn.Dense(84, num_classes)
        ])

    def construct(self, x: Tensor):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


@register_model
def LeNet5(pretrained: bool = False, in_channels: int = 1, num_classes: int = 10, **kwargs):
    default_cfg = default_cfgs["LeNet5"]
    model = LeNet(in_channels=in_channels, num_classes=num_classes)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model