######################################################################
#  script name  : CategoricalConditionalBatchNorm2d.py
#  author       : Chen Xuanhong
#  created time : 2019/9/12 00:07
#  modification time ：2019/9/12 00:07
#  modified by  : Chen Xuanhong
######################################################################

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from ConditionalBatchNorm2d import ConditionalBatchNorm2d

class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalConditionalBatchNorm2d, self).forward(input, weight, bias)