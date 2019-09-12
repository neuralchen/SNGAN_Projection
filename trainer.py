######################################################################
#  script name  : trainer.py
#  author       : Chen Xuanhong
#  created time : 2019/9/11 22:36
#  modification time ：2019/9/11 22:36
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
from utilities.DistributionClass import prepare_z
from utilities.Reporter import Reporter
import utilities.Sampler as Sampler

from metric.FID import *
from torchviz import make_dot
from cifar10_model_ws import Generator, Discriminator
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
        self.g_num      = config.g_num
        self.z_dim      = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel   = config.parallel
        self.seed       = config.seed

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
        self.metric_caculation_step = config.metric_caculation_step
        self.version        = config.version
        self.caculate_FID   = config.caculate_FID

        # Path
        self.log_path       = os.path.join(config.log_path, self.version)
        self.summary_path   = self.log_path
        self.sample_path    = os.path.join(config.sample_path, self.version)
        self.model_save_path= os.path.join(config.model_save_path, self.version)
        
        self.build_model()
        self.reporter.writeConfig(config)
        # with open(self.report_file,'w') as logf:
        #     logf.writelines("The training process is start from %s\n"%datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S'))
        #     i = 1
        #     for item in config.__dict__.items():
        #         text = "[%d] %s--%s\n"%(i,item[0],str(item[1]))
        #         logf.writelines(text)
        #         i+=1
        self.reporter.writeModel(self.G.__str__())
        self.reporter.writeModel(self.D.__str__())
            # logf.writelines(self.G.__str__()+"\n")
            # logf.writelines(self.D.__str__()+"\n")
        # Metric noise sampling
        if self.caculate_FID:
            z_sampler = prepare_z(self.batch_size,self.z_dim)
            sample = functools.partial(Sampler.sample,G=self.G,z_=z_sampler, parallel=self.parallel)
            self.get_inception_metrics = prepare_inception_metrics(config.FID_mean_cov,sample,config.num_inception_images)

        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.summary_path)
            z = tensor2var(torch.randn(self.batch_size, self.z_dim))
            # self.writer.add_graph(self.G,z)
            y = tensor2var(torch.randn(self.batch_size,3,self.imsize,self.imsize))
            # self.writer.add_graph(self.D,y)
            vise_graph = make_dot(self.G(z), params=dict(self.G.named_parameters()))
            vise_graph.view(self.log_path+"/Generator")
            vise_graph = make_dot(self.D(y), params=dict(self.D.named_parameters()))
            vise_graph.view(self.log_path+"/Discriminator")
            del z
            del y
            # self.writer.add_graph(self.D)

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
       
        # step_per_epoch = len(self.data_loader)

        model_save_step = self.model_save_step

        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
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
                real_images, _ = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images, _ = next(data_iter)

            # Compute loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            real_images = tensor2var(real_images)
            # d_out_real,dr1,dr2 = self.D(real_images)
            d_out_real = self.D(real_images)
           
            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
                
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
                
                         
            # apply Gumbel Softmax
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            # fake_images,gf1,gf2 = self.G(z)
            fake_images = self.G(z)
            
            # d_out_fake,df1,df2 = self.D(fake_images)
            d_out_fake = self.D(fake_images)

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()

            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()


            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()


            if self.adv_loss == 'wgan-gp':
                # Compute gradient penalty
                alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
                out,_,_ = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
            
            # ================== Train G and gumbel ================== #
            # Create random noise
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            # fake_images,_,_ = self.G(z)
            fake_images = self.G(z)

            # Compute loss with fake images
            # g_out_fake,_,_ = self.D(fake_images)  # batch x n
            g_out_fake = self.D(fake_images)
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
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
                fake_images = self.G(fixed_z)

                save_image(denorm(fake_images.data),

                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))



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

    def build_model(self):

        self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).cuda()
        self.D = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        # print(self.G)
        # print(self.D)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))