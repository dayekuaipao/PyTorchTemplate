import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std
    def __call__(self,image):        
        image = np.array(image).astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image

class ToTensor(object):
    """Convert ndarrays in image to Tensors."""

    def __call__(self,image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W        
        image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        return image


class RandomHorizontalFlip(object):
    def __call__(self,image):         
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image


class RandomRotate(object):
    def __init__(self, degree,interpolation=Image.BILINEAR):
        self.degree = degree
        self.interpolation = interpolation
    def __call__(self,image):        
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        image = image.rotate(rotate_degree, self.interpolation)

        return image


class RandomGaussianBlur(object):
    def __call__(self,image):          
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return image


class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self,image):    
        w, h = image.size
        th, tw = self.crop_size
        if w == tw and h == th:
            return image.crop(0, 0, h, w)
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)  
            return image.crop((i, j, i+th, j+tw))

class Pad(object):
    def __init__(self,padding,fill=0):
        self.padding=padding
        self.fill=fill

    def __call__(self,image):
        return ImageOps.expand(image, border=self.padding, fill=self.fill)

class PadFixedSize(object):
    def __init__(self,size):
        self.size=size
    def __call__(self,image):
        img = np.array(image)
        if len(img.shape) == 3:
            img = np.pad(img, ((0, self.size[0]-img.shape[0]), (0, self.size[1]-img.shape[1]), (0, 0)))
        if len(img.shape) == 2:
            img = np.pad(img, ((0, self.size[0]-img.shape[0]), (0, self.size[1]-img.shape[1])))
        image = Image.fromarray(img)
        return image

class PadOrCropFixedSize(object):
    def __init__(self,size):
        self.size=size
    def __call__(self,image,label):
        img = np.array(image)
        w, h = image.size
        th, tw = self.size    
        if w>tw and h>th:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw) 
            return image.crop((i, j, i+th, j+tw))
        else:
            if w<tw:
                if len(img.shape) == 3:
                    img = np.pad(img, ((0, tw-w), (0, 0), (0, 0)))
                if len(img.shape) == 2:
                    img = np.pad(img, ((0, tw-w), (0, 0)))
            if h<th:
                if len(img.shape) == 3:
                    img = np.pad(img, ((0, th-h), (0, 0), (0, 0)))
                if len(img.shape) == 2:
                    img = np.pad(img, ((0, th-h), (0, 0)))
            return image.crop((0, 0, th, tw))

class CenterCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size
    def __call__(self,image):  
        w, h = image.size
        th, tw = self.crop_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return image.crop(image, i, j, th, tw)      

class Resize(object):
    def __init__(self, size,interpolation = Image.BILINEAR):
        self.size = size  # size: (h, w)
        self.interpolation = interpolation
    def __call__(self,image): 
        image = image.resize(self.size, self.interpolation)
        return image

class RandomScale (object):
    def  __init__(self, scale ,interpolation = Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation
    def __call__(self,image):
        scale = random.random()*(self.scale[1]-self.scale[0])+self.scale[0]
        image = image.resize((int(image.size[0]*scale),int(image.size[1]*scale)),self.interpolation)
        return image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image