######################################################################
#  script name  : parameter.py
#  author       : Chen Xuanhong
#  created time : 2019/9/11 22:36
#  modification time ：2019/9/11 22:36
#  modified by  : Chen Xuanhong
######################################################################

import argparse

def str2bool(v):
    return v.lower() in ('true')

def getParameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='sagan', choices=['sagan', 'qgan'])
    parser.add_argument('--adv_loss', type=str, default='hinge', choices=['wgan-gp', 'hinge'])
    parser.add_argument('--imsize', type=int, default=32)
    parser.add_argument('--g_num', type=int, default=5)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--version', type=str, default='sagan_cifar10_6')

    # Training setting
    parser.add_argument('--total_step', type=int, default=1000000, help='how many times to update the generator')
    parser.add_argument('--d_iters', type=float, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['lsun', 'celeba','cifar10'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--experiment_description', type=str, default="测试ws加入了仿射系数，测试其效果")
    # Path
    parser.add_argument('--image_path', type=str, default='./data')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--attn_path', type=str, default='./attn')
    parser.add_argument('--FID_mean_cov', type=str, default='./datasetMoment/cifar10/')

    # Step size
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=5000)
    parser.add_argument('--metric_caculation_step', type=int, default=2000)
    parser.add_argument('--caculate_FID', type=str2bool, default=True)
    parser.add_argument('--num_inception_images', type=int, default=200)
    parser.add_argument('--test_images_num', type=int, default=200)

    return parser.parse_args()