######################################################################
#  script name  : Utilities.py
#  author       : Chen Xuanhong
#  created time : 2019/9/11 22:36
#  modification time ：2019/9/11 22:36
#  modified by  : Chen Xuanhong
######################################################################

import  os
import  torch
from    torch.autograd import Variable
import  numpy as np


def makeFolder(path, version):
        if not os.path.exists(os.path.join(path, version)):
            os.makedirs(os.path.join(path, version))


def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

def var2tensor(x):
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)