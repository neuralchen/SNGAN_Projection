######################################################################
#  script name  : main.py
#  author       : Chen Xuanhong
#  created time : 2019/9/11 22:36
#  modification time ：2019/9/12 21:36
#  modified by  : Chen Xuanhong
######################################################################

from    parameter import *
from    trainer import Trainer
# from    tester import Tester
from    dataTool.dataLoader import DataLoader
from    torch.backends import cudnn
from    utilities.Utilities import makeFolder
import  torch

def main(config):
    # For fast training
    cudnn.benchmark = True


    # Data loader
    data_loader = DataLoader(config.train, config.dataset, config.image_path, config.imsize,
                             config.batch_size, shuf=config.train)

    # Create directories if not exist
    makeFolder(config.model_save_path, config.version)
    makeFolder(config.sample_path, config.version)
    makeFolder(config.log_path, config.version)


    if config.train:
        trainer = Trainer(data_loader.loader(), config)
        trainer.train()
    else:
        tester = Tester(data_loader.loader(), config)
        tester.test()

if __name__ == '__main__':
    config = getParameters()
    main(config)