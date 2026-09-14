"""
DEX (Deep EXpectation of apparent age)[1] のVGG16ベースの年齢・性別推定モデル定義．

評価指標(年齢の一致度・性別の一致率)の算出にのみ使用し，匿名化の推論パイプライン本体
(`identity_anonymizer.faceswap`)からは参照しない．

[1] R. Rothe et al., "Dex: Deep expectation of apparent age from a single image,"
    Proc. IEEE CVPR workshops, December 2015.
"""

from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F


def _vgg_block(in_channels: int, out_channels: int, more: bool = False) -> nn.Sequential:
    layers = [
        ("conv1", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
        ("relu1", nn.ReLU(inplace=True)),
        ("conv2", nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
        ("relu2", nn.ReLU(inplace=True)),
    ]
    if more:
        layers.extend([
            ("conv3", nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
        ])
    layers.append(("maxpool", nn.MaxPool2d(kernel_size=2, stride=2)))
    return nn.Sequential(OrderedDict(layers))


class _VGG(nn.Module):
    def __init__(self, classes: int = 1000, channels: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            _vgg_block(channels, 64),
            _vgg_block(64, 128),
            _vgg_block(128, 256, True),
            _vgg_block(256, 512, True),
            _vgg_block(512, 512, True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=True),
        )
        self.cls = nn.Linear(4096, classes)

    def forward(self, x):
        in_size = x.shape[0]
        x = self.conv(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.cls(x)
        return F.softmax(x, dim=1)


class DexGenderModel(_VGG):
    """性別を2クラス(女性/男性)で推定するDEXモデル．"""

    def __init__(self):
        super().__init__(classes=2)


class DexAgeModel(_VGG):
    """見た目年齢を0〜100歳の101クラスで推定するDEXモデル．"""

    def __init__(self):
        super().__init__(classes=101)
