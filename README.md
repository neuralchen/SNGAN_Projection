# GANs with spectral normalization and projection discriminator
This is an unofficial PyTorch implementation of [sngan_projection](https://github.com/pfnet-research/sngan_projection)

[Miyato, Takeru, and Masanori Koyama. "cGANs with projection discriminator." arXiv preprint arXiv:1802.05637 (2018).](https://arxiv.org/abs/1802.05637)

# Dependencies:
- PyTorch1.0
- numpy
- scipy
- tensorboardX
- tqdm
- [torchviz](https://github.com/szagoruyko/pytorchviz) pip install torchviz and [graphviz](http://www.graphviz.org/) sudo apt-get install graphviz

# Usage:
There are two to run the training script:
- Run the script directly (We recommend this way): `python3 main.py` or `python main.py`.
    In this way, the training parameters can be modified by modifying the `parameter.py` parameter defaults.

# Parameters
|  Parameters   | Function  |
|  :----  | :----  |
| --version  | Experiment name |
| --train  | Set the model stage, Ture---training stage; False---testing stage |
| --experiment_description  | Descriptive text for this experiment  |
| --total_step  | Totally training step |
| --batch_size  | Batch size |
| --g_lr  | Learning rate of generator |
| --d_lr  | Learning rate of discriminator |
| --parallel  | Enable the parallel training |
| --dataset  | Set the dataset name,lsun,celeb,cifar10 |
| --cuda  | Set GPU device number |
| --image_path  | The root dir to training dataset |
| --FID_mean_cov  | The root dir to dataset moments npz file |

# Results
We have reproduced the FID (in Cifar-10, best result is FID=17.2) result reported in the paper.

The convergence curve of FID is as follows:

![image](https://github.com/XHChen0528/SNGAN_Projection_Pytorch/blob/master/figures/fid_result.JPG)

## CIFAR10 results
200K:

![image](https://github.com/XHChen0528/SNGAN_Projection_Pytorch/blob/master/figures/200000_fake.png)

500K:

![image](https://github.com/XHChen0528/SNGAN_Projection_Pytorch/blob/master/figures/500000_fake.png)

600K:

![image](https://github.com/XHChen0528/SNGAN_Projection_Pytorch/blob/master/figures/600000_fake.png)

800K:

![image](https://github.com/XHChen0528/SNGAN_Projection_Pytorch/blob/master/figures/800000_fake.png)

1000K:

![image](https://github.com/XHChen0528/SNGAN_Projection_Pytorch/blob/master/figures/1000000_fake.png)





# Acknowledgement
- [sngan_projection](https://github.com/pfnet-research/sngan_projection)
- [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch)
- [pytorch.sngan_projection](https://github.com/crcrpar/pytorch.sngan_projection)