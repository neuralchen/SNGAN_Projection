''' Inception utilities
    This file contains methods for calculating IS and FID, using either
    the original numpy code or an accelerated fully-pytorch version that 
    uses a fast newton-schulz approximation for the matrix sqrt. There are also
    methods for acquiring a desired number of samples from the Generator,
    and parallelizing the inbuilt PyTorch inception network.
    
    NOTE that Inception Scores and FIDs calculated using these methods will 
    *not* be directly comparable to values calculated using the original TF
    IS/FID code. You *must* use the TF model if you wish to report and compare
    numbers. This code tends to produce IS values that are 5-10% lower than
    those obtained through TF. 
'''    
import numpy as np
import time
from scipy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
from torchvision.models.inception import inception_v3
from tqdm import tqdm
# from utilities.Decorator import time_it




# Module that wraps the inception network to enable use with dataparallel and
# returning pool features and logits.
class WrapInception(nn.Module):
  def __init__(self, net):
    super(WrapInception,self).__init__()
    self.net = net
    self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                  requires_grad=False)
    self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                 requires_grad=False)
  def forward(self, x):
    # Normalize x
    x = (x + 1.) / 2.0
    x = (x - self.mean) / self.std
    # Upsample if necessary
    if x.shape[2] != 299 or x.shape[3] != 299:
      x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
    # 299 x 299 x 3
    x = self.net.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = self.net.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = self.net.Conv2d_2b_3x3(x)
    # 147 x 147 x 64
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 73 x 73 x 64
    x = self.net.Conv2d_3b_1x1(x)
    # 73 x 73 x 80
    x = self.net.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 35 x 35 x 192
    x = self.net.Mixed_5b(x)
    # 35 x 35 x 256
    x = self.net.Mixed_5c(x)
    # 35 x 35 x 288
    x = self.net.Mixed_5d(x)
    # 35 x 35 x 288
    x = self.net.Mixed_6a(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6b(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6c(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6d(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6e(x)
    # 17 x 17 x 768
    # 17 x 17 x 768
    x = self.net.Mixed_7a(x)
    # 8 x 8 x 1280
    x = self.net.Mixed_7b(x)
    # 8 x 8 x 2048
    x = self.net.Mixed_7c(x)
    # 8 x 8 x 2048
    pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
    # 1 x 1 x 2048
    # logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
    # 1000 (num_classes)
    return pool#, logits


# A pytorch implementation of cov, from Modar M. Alfadly
# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt 
def sqrt_newton_schulz(A, numIters, dtype=None):
  with torch.no_grad():
    if dtype is None:
      dtype = A.type()
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
    I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    for i in range(numIters):
      T = 0.5*(3.0*I - Z.bmm(Y))
      Y = Y.bmm(T)
      Z = T.bmm(Z)
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
  return sA

# @time_it
def torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Pytorch implementation of the Frechet Distance.
  Taken from https://github.com/bioinf-jku/TTUR
  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
  and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
  Stable version by Dougal J. Sutherland.
  Params:
  -- mu1   : Numpy array containing the activations of a layer of the
             inception net (like returned by the function 'get_predictions')
             for generated samples.
  -- mu2   : The sample mean over activations, precalculated on an 
             representive data set.
  -- sigma1: The covariance matrix over activations for generated samples.
  -- sigma2: The covariance matrix over activations, precalculated on an 
             representive data set.
  Returns:
  --   : The Frechet Distance.
  """


  assert mu1.shape == mu2.shape, \
    'Training and test mean vectors have different lengths'
  assert sigma1.shape == sigma2.shape, \
    'Training and test covariances have different dimensions'

  # diff = mu1 - mu2
  # # Run 50 itrs of newton-schulz to get the matrix sqrt of sigma1 dot sigma2
  # covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50).squeeze()
  # 
  # covmean = sqrt_newton_schulz(mulRes.unsqueeze(0), 50).squeeze()
  # res = torch.sqrt(mulRes)
  # c = torch.trace(res)
  # a = torch.trace(sigma1)
  # b = torch.trace(sigma2)
  # # out = (diff.dot(diff) +  torch.trace(sigma1) + torch.trace(sigma2)
  # #        - 2 * torch.trace(covmean))
  # out = diff.dot(diff) +  a + b - 2 * c
  diff = mu1 - mu2
  diff = diff.dot(diff)
  mulRes = sigma1.mm(sigma2) 
  covmean, _ = linalg.sqrtm(mulRes.cpu(), disp=False)
  if not np.isfinite(covmean).all():
      msg = ('fid calculation produces singular product; '
              'adding %s to diagonal of cov estimates') % eps
      print(msg)
      offset = np.eye(sigma1.shape[0]) * eps
      covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # Numerical error might give slight imaginary component
  if np.iscomplexobj(covmean):
      if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
          m = np.max(np.abs(covmean.imag))
          raise ValueError('Imaginary component {}'.format(m))
      covmean = covmean.real
  # covmean = covmean.cuda()
  # tr_covmean = np.trace(covmean)
  out = diff.cpu() + np.trace(sigma1.cpu())+np.trace(sigma2.cpu()) - 2 *np.trace(covmean)
  return out

# @time_it
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Numpy implementation of the Frechet Distance.
  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
  and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

  Stable version by Dougal J. Sutherland.

  Params:
  -- mu1   : Numpy array containing the activations of a layer of the
              inception net (like returned by the function 'get_predictions')
              for generated samples.
  -- mu2   : The sample mean over activations, precalculated on an
              representative data set.
  -- sigma1: The covariance matrix over activations for generated samples.
  -- sigma2: The covariance matrix over activations, precalculated on an
              representative data set.

  Returns:
  --   : The Frechet Distance.
  """

  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)

  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)

  assert mu1.shape == mu2.shape, \
      'Training and test mean vectors have different lengths'
  assert sigma1.shape == sigma2.shape, \
      'Training and test covariances have different dimensions'

  diff = mu1 - mu2

  # Product might be almost singular
  covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
      msg = ('fid calculation produces singular product; '
              'adding %s to diagonal of cov estimates') % eps
      print(msg)
      offset = np.eye(sigma1.shape[0]) * eps
      covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # Numerical error might give slight imaginary component
  if np.iscomplexobj(covmean):
      if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
          m = np.max(np.abs(covmean.imag))
          raise ValueError('Imaginary component {}'.format(m))
      covmean = covmean.real

  tr_covmean = np.trace(covmean)

  return (diff.dot(diff) + np.trace(sigma1) +
          np.trace(sigma2) - 2 * tr_covmean)

# Loop and run the sampler and the net until it accumulates num_inception_images
# activations. Return the pool, the logits, and the labels (if one wants 
# Inception Accuracy the labels of the generated class will be needed)
def accumulate_inception_activations(sample, net, num_inception_images=50000,isDataLoader = False):
  # pool, logits, labels = [], [], []
  pool = []
  for i in tqdm(range(num_inception_images)):

  # while (torch.cat(logits, 0).shape[0] if len(logits) else 0) < num_inception_images:
    with torch.no_grad():
      if isDataLoader:
        images, _ = next(sample)
        images = images.cuda()
      else:
        images = sample()
      # pool_val, logits_val = net(images.float())
      pool_val = net(images.float())
      pool += [pool_val] 
      # logits += [F.softmax(logits_val, 1)]
      # labels += [labels_val]
  return torch.cat(pool, 0)#, torch.cat(logits, 0), torch.cat(labels, 0)


# Load and wrap the Inception model
def load_inception_net(parallel=False):
  inception_model = inception_v3(pretrained=True, transform_input=False)
  inception_model = WrapInception(inception_model.eval()).cuda()
  if parallel:
    print('Parallelizing Inception module...')
    inception_model = nn.DataParallel(inception_model)
  return inception_model


# This produces a function which takes in an iterator which returns a set number of samples
# and iterates until it accumulates config['num_inception_images'] images.
# The iterator can return samples with a different batch size than used in
# training, using the setting confg['inception_batchsize']
def prepare_inception_metrics(dataset, sampler,num_inception_images, parallel=False):
  # Load metrics; this is intentionally not in a try-except loop so that
  # the script will crash here if it cannot find the Inception moments.
  # By default, remove the "hdf5" from dataset
  data_mu     = np.load(dataset+'C10_inception_moments.npz')['mu']
  data_sigma  = np.load(dataset+'C10_inception_moments.npz')['sigma']
  # Load network
  net = load_inception_net(parallel)
  def get_inception_metrics():
    print('Gathering activations...')
    pool      = accumulate_inception_activations(sampler, net, num_inception_images)
    print('Calculating means and covariances...')
    mu, sigma = torch.mean(pool, 0), torch_cov(pool, rowvar=False)
    print('Covariances calculated, getting FID...')
    FID       = calculate_frechet_distance(mu.cpu(),sigma.cpu(),data_mu,data_sigma)
    FID       = float(FID)
    del mu, sigma
    return FID
  return get_inception_metrics

def prepare_dataset_inception_moments(dataRoot,savePath,dataName="cifar10"):
  batchSize = 50
  norm_mean = [0.5,0.5,0.5]
  norm_std = [0.5,0.5,0.5]
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(norm_mean, norm_std)])
  trainset = torchvision.datasets.CIFAR10(root=dataRoot+'/cifar10', train=True,
                                        download=True, transform=transform)
                                      
  loader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=batchSize,
                                              shuffle=False,
                                              num_workers=8,
                                              drop_last=True)
  net = load_inception_net()
  totalNum = len(loader)
  print("total batch number:%s"%totalNum)
  # totalNum =5
  data_iter = iter(loader)

  # for i in tqdm(range(totalNum)):
  #   batch, _ = next(data_iter)
  pool = accumulate_inception_activations(data_iter, net, totalNum,True)
  pool = pool.cpu().numpy()
  mu, sigma = np.mean(pool, 0), np.cov(pool, rowvar=False)
  print("mean of the dataset:/n")
  print(mu)
  print(mu.shape)
  
  print("std of the dataset:/n")
  print(sigma)
  print(sigma.shape)
  np.savez(savePath+'/cifar10/C10_inception_moments.npz', mu=mu, sigma=sigma)

def caculate_inception_metrics(dataRoot, momentsPath):
  # Load metrics; this is intentionally not in a try-except loop so that
  # the script will crash here if it cannot find the Inception moments.
  # By default, remove the "hdf5" from dataset
  batchSize = 50
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  trainset = torchvision.datasets.CIFAR10(root=dataRoot+'/cifar10', train=False,
                                        download=True, transform=transform)
                                      
  loader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=batchSize,
                                              shuffle=False,
                                              num_workers=8,
                                              drop_last=True)
  net = load_inception_net()
  totalNum = len(loader)
  # totalNum = 200
  print("total batch number:%s"%totalNum)
  # totalNum =5
  data_iter = iter(loader)

  data_mu = np.load(momentsPath+'_inception_moments.npz')['mu']
  data_sigma = np.load(momentsPath+'_inception_moments.npz')['sigma']
  # Load network
  net = load_inception_net()
  print('Gathering activations...')
  pool = accumulate_inception_activations(data_iter, net, totalNum,True)
  print('Calculating means and covariances...')
  mu, sigma = torch.mean(pool, 0), torch_cov(pool, rowvar=False)
  print('Covariances calculated, getting FID...')
  # FID = torch_calculate_frechet_distance(mu, sigma, torch.tensor(data_mu).float().cuda(), torch.tensor(data_sigma).float().cuda())
  FID = calculate_frechet_distance(mu.cpu(),sigma.cpu(),data_mu,data_sigma)
  FID = float(FID)
  print(FID)

if __name__ == "__main__":
    prepare_dataset_inception_moments("./data",'./datasetMoment')