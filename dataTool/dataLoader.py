import torch
import torchvision.datasets as dsets
from torchvision import transforms


class DataLoader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        # torch.multiprocessing.set_start_method('spawn')
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            cropSize = min(self.imsize,160)
            options.append(transforms.CenterCrop(cropSize))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def loadLsun(self, classes='church_outdoor_train'):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.LSUN(self.path, classes=[classes], transform=transforms)
        return dataset

    def loadCeleb(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/img_align_celeba', transform=transforms)
        return dataset
    
    def loadCifar10(self):
        transforms = self.transform(True, True, True, True)
        trainset = dsets.CIFAR10(root=self.path+'/cifar10', train=True,
                                        download=True, transform=transforms)
        return trainset


    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.loadLsun()
        elif self.dataset == 'celeba':
            dataset = self.loadCeleb()
        elif self.dataset == 'cifar10':
            dataset = self.loadCifar10()

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=4,
                                              drop_last=True)
        return loader

