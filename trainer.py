######################################################################
#  script name  : trainer.py
#  author       : Chen Xuanhong
#  created time : 2019/9/11 22:36
#  modification time ：2019/9/13 11:35
#  modified by  : Chen Xuanhong
######################################################################

import os
import time
import datetime
import functools

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from utilities.Utilities import *
from tensorboardX import SummaryWriter
from utilities.Reporter import Reporter
import utilities.Sampler as Sampler

from torchviz import make_dot
from components.GenResNet32 import ResNetGenerator
from components.SNResNetProjectionDiscriminator32 import SNResNetProjectionDiscriminator
import metrics.FID as FIDCaculator
#from inceptionScoreMetricClass import inceptionScoreMetricClass


class Trainer(object):
    def __init__(self, data_loader, config):

        self.report_file = os.path.join(config.log_path, config.version,config.version+"_report.log")
        self.reporter = Reporter(self.report_file)

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.cGAN = config.cGAN
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize     = config.imsize
        self.z_dim      = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.n_classes  = config.n_class if config.cGAN else 0
        self.parallel   = config.parallel
        self.seed       = config.seed
        self.device     = torch.device('cuda:%d'%config.cuda)
        self.GPUs       = config.GPUs

        self.gen_distribution = config.gen_distribution
        self.gen_bottom_width = config.gen_bottom_width

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers= config.num_workers
        self.g_lr       = config.g_lr
        self.d_lr       = config.d_lr
        self.lr_decay   = config.lr_decay
        self.beta1      = config.beta1
        self.beta2      = config.beta2

        self.use_pretrained_model   = config.use_pretrained_model
        self.chechpoint_step        = config.chechpoint_step
        self.use_pretrained_model   = config.use_pretrained_model

        self.dataset        = config.dataset
        self.use_tensorboard= config.use_tensorboard
        self.image_path     = config.image_path
        self.log_path       = config.log_path
        self.model_save_path= config.model_save_path
        self.sample_path    = config.sample_path
        self.log_step       = config.log_step
        self.sample_step    = config.sample_step
        self.model_save_step= config.model_save_step
        self.version        = config.version
        self.caculate_FID   = config.caculate_FID

        self.metric_caculation_step = config.metric_caculation_step

        # Path
        self.log_path       = os.path.join(config.log_path, self.version)
        self.summary_path   = self.log_path
        self.sample_path    = os.path.join(config.sample_path, self.version)
        self.model_save_path= os.path.join(config.model_save_path, self.version)
        
        self.build_model()
        self.reporter.writeConfig(config)
        self.reporter.writeModel(self.G.__str__())
        self.reporter.writeModel(self.D.__str__())
        if self.caculate_FID:
            z_sampler,c_sampler = Sampler.prepare_z_c(self.batch_size,self.z_dim,self.n_classes,device=self.device)
            gsampler = functools.partial(Sampler.sampleG,G=self.G,z_=z_sampler,c_=c_sampler, parallel=self.parallel)
            self.get_inception_metrics = FIDCaculator.prepare_inception_metrics(config.FID_mean_cov,gsampler,config.metric_images_num)

        self.writer = SummaryWriter(log_dir=self.summary_path)
        z = torch.zeros(1, self.z_dim).to(self.device)
        c = torch.zeros(1).long().to(self.device)
        y = torch.zeros(1,3,self.imsize,self.imsize).to(self.device)
        vise_graph = make_dot(self.G(z,c), params=dict(self.G.named_parameters()))
        vise_graph.view(self.log_path+"/Generator")
        vise_graph = make_dot(self.D(y,c), params=dict(self.D.named_parameters()))
        vise_graph.view(self.log_path+"/Discriminator")
        del z
        del c
        del y
            # self.writer.add_graph(self.D)

        # Start with trained model
        if self.use_pretrained_model:
            self.load_pretrained_model()
    
    def build_model(self):
        self.G = ResNetGenerator(self.g_conv_dim, self.z_dim, self.gen_bottom_width,
                            num_classes=self.n_classes).to(self.device)
        self.D = SNResNetProjectionDiscriminator(self.d_conv_dim, self.n_classes).to(self.device)
        if self.parallel:
            self.G = nn.DataParallel(self.G,device_ids=self.GPUs)
            self.D = nn.DataParallel(self.D,device_ids=self.GPUs)
        # Loss and optimizer
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        model_save_step = self.model_save_step

        # Fixed input for debugging
        sampleBatch = 10
        fixed_z = torch.randn(self.n_classes*sampleBatch, self.z_dim)
        fixed_z = fixed_z.to(self.device)
        fixed_c = Sampler.sampleFixedLabels(self.n_classes,sampleBatch,self.device)

        runingZ,runingLabel = Sampler.prepare_z_c(self.batch_size, self.z_dim, self.n_classes, device=self.device)

        # Start with trained model
        if self.use_pretrained_model:
            start = self.chechpoint_step + 1
        else:
            start = 0
        # Start time
        start_time = time.time()
        self.reporter.writeInfo("Start to train the model")
        for step in range(start, self.total_step):
            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            try:
                realImages, realLabel = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                realImages, realLabel = next(data_iter)

            # Compute loss with real images
            realImages  = realImages.to(self.device)
            realLabel   = realLabel.to(self.device).long()
            d_out_real  = self.D(realImages,realLabel)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax
            runingZ.sample_()
            runingLabel.sample_()
            fake_images = self.G(runingZ,runingLabel)
            d_out_fake  = self.D(fake_images,runingLabel)
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            # Backward + Optimize
            d_loss      = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            
            # ================== Train G and gumbel ================== #
            # Create random noise
            runingZ.sample_()
            runingLabel.sample_()
            fake_images = self.G(runingZ,runingLabel)

            # Compute loss with fake images
            g_out_fake  = self.D(fake_images,runingLabel)
            g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()


            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                      " d_loss_fake: {:.4f}, g_loss_fake: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step , d_loss_real.item(),
                             d_loss_fake.item(), g_loss_fake.item()))
                
                   
                self.writer.add_scalar('log/d_loss_real', d_loss_real.item(),(step + 1))
                self.writer.add_scalar('log/d_loss_fake', d_loss_fake.item(),(step + 1))
                self.writer.add_scalar('log/d_loss', d_loss.item(), (step + 1))
                self.writer.add_scalar('log/g_loss_fake', g_loss_fake.item(), (step + 1))
 
            if (step + 1) % self.sample_step == 0:
                fake_images = self.G(fixed_z,fixed_c)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)),nrow=self.n_classes)

            if (step+1) % model_save_step==0:

                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))

                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))
            
            if (step+1) % self.metric_caculation_step == 0 and self.caculate_FID:
                print("start to caculate the FID")
                FID = self.get_inception_metrics()
                print("FID is %.3f"%FID)
                self.writer.add_scalar('metric/FID', FID, (step + 1))
                self.reporter.writeTrainLog(step+1,"Current FID is %.4f"%FID)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.chechpoint_step))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.chechpoint_step))))
        print('loaded trained models (step: {})..!'.format(self.chechpoint_step))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()