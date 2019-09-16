######################################################################
#  script name  : GenResNet32.py
#  author       : Chen Xuanhong
#  created time : 2019/9/12 00:24
#  modification time ：2019/9/12 00:24
#  modified by  : Chen Xuanhong
######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from components.GenResBlock import GenResBlock

class ResNetGenerator(nn.Module):
    """Generator generates 32x32."""

    def __init__(self, num_features=64, dim_z=128, bottom_width=4, num_classes=0,
                 activation=F.relu):
        super(ResNetGenerator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes

        self.l1 = nn.Linear(dim_z, 8 * num_features * bottom_width ** 2)

        # self.block2 = GenResBlock(num_features * 16, num_features * 8,
        #                     activation=activation, upsample=True,
        #                     num_classes=num_classes)
        self.block2 = GenResBlock(num_features * 8, num_features * 4,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block3 = GenResBlock(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block4 = GenResBlock(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.b5     = nn.BatchNorm2d(num_features)
        self.conv5  = nn.Conv2d(num_features, 3, 1, 1)
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.data)
        init.xavier_uniform_(self.conv7.weight.data)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 5):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b5(h))
        return torch.tanh(self.conv5(h))