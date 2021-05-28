
import os

from functools import partial

import numpy as np

import chainer
from chainer.iterators import SerialIterator

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100
from chainer.datasets import TransformDataset

from chainer.backends import cuda

from skimage import transform as skimage_transform
from chainercv import transforms


# Keep a cleaned up model namespace
from models import models
    

def add_base_arguments(parser, models=models):
    
    parser.add_argument('--model', '-m', choices=models.keys(), default='resnet50',
                        help='Convnet architecture')
    
    parser.add_argument('--dataset', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    
    parser.add_argument('--augment','-a', default=2, type=int, choices=[0,1,2],
                        help='Augment level')
    
    parser.add_argument('--epoch', '-e', type=float, default=300,
                        help='Number of epochs to train (default: 300)')
    
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Learning minibatch size (default: 128)')
    
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Initial learning rate for SGD (default: 0.05)')
    
    parser.add_argument('--learnrate_decay', '-y', type=float, default=70,
                        help='Number of epochs to decrease learning rate after (default: 70)')
    
    parser.add_argument('--momentum', '-p', type=float, default=0.9,
                        help='Momentum for SGD')
    
    parser.add_argument('--weight_decay', '-w', type=float, default=5e-4,
                        help='Weight decay per epoch (default=5e-4)')
    
    parser.add_argument('--seed','-s', type=int, default=None,
                        help='Initial seeding for random numbers')
    
    parser.add_argument('--device', '-d', type=int, default=0,
                        help='GPU ID (negative value indicates CPU')
    
    parser.add_argument('--snapshot_every', metavar='EPOCHS', type=int,
                        help='Number of epochs to take a snapshot after')
    
    parser.add_argument('--update_interval','-u', default=20, type=int,
                        help='Progress bar update interval (iterations)')
    
    parser.add_argument('--log_interval', type=float, default=None,
                        help='Interval for logging and progressbar updating')
    
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    

def load_dataset(batchsize, dataset, augment=2):
    
    scale = 255.0 if augment else 1.0
    if dataset == 'cifar10':
        train, test = get_cifar10(scale=scale)
    elif dataset == 'cifar100':
        train, test = get_cifar100(scale=scale)
    else:
        raise Exception('Invalid dataset choice {}.'.format(dataset))
    
    if augment:
        #mean = np.mean(train._datasets[0], axis=(0, 2, 3))
        #std = np.std(train._datasets[0], axis=(0, 2, 3))
        # Pre calculated from above
        mean = np.array([125.3, 123.0, 113.9])
        std = np.array([  63.0,  62.1,  66.7])
        
        train = normalize_dataset(train, mean, std)
        test = normalize_dataset(test, mean, std)
        
        # Previously pca was 25.5 or 10% of 255
        # Now we normalize, so to keep PCA at 10% of the range we use the min and max of the 
        # normalized datasets
        
        #pca_sigma = 0.1 * (np.max(train._datasets[0] - np.min(train._datasets[0])
        # Pre calculated from above
        pca_sigma = 0.1 * ((2.12) - (-1.99))  # = 0.4116
        
        
        if augment == 1:
            train = pad_dataset(train, pad=4)
            kwargs = dict(crop_size=(32, 32), cutout=8, flip=True)
            train_transform = partial(transform_fast, **kwargs)
            test_transform = lambda x:x # No augmentation
        elif augment == 2:
            kwargs = dict(crop_size=(32, 32), expand_ratio=1.2, 
                          pca_sigma=pca_sigma, random_angle=15.0, 
                          train=True, cutout=8, flip=True)
            train_transform = partial(transform, **kwargs)
            test_transform = partial(transform, train=False)
    
        train = TransformDataset(train, train_transform)
        test = TransformDataset(test, test_transform)
    
    
    train_iter = SerialIterator(train, batchsize)
    test_iter = SerialIterator(test, batchsize, repeat=False, shuffle=False)
    
    return train_iter, test_iter
    

def init_model(model, device=-1, dataset='cifar10', dtype=np.float32):
    """ Initializes the model to train and (optionally) sends it to the gpu """
    if dtype == np.float32:
        model = model()
    else:
        model = model(dtype=dtype)
    
    
    print('Using device {}'.format(device))
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()  # Make the GPU current
        model.to_gpu()
        
    return model

    
def _linear_congruential_rng(seed):
    ''' glibc's linear congruential generator'''
    while True:
        seed = int( (seed*1103515245 + 12345)%(2**31-1) ) if seed is not None else None
        yield seed
        
def seed_rng(seed, gpu=-1):
    if seed is None:
        return None, None
    
    os.environ['CHAINER_SEED'] = str(seed)
    
    rng = _linear_congruential_rng(seed)
    next(rng) # Make sure we don't use the same seed as for main chainer rng
        
    fixed_seeds = (next(rng), next(rng))
    if gpu >= 0 and seed is not None:
        reseed_rng(fixed_seeds)
    
    return rng, fixed_seeds

def reseed_rng(fixed_seeds):
    if fixed_seeds is not None:
        cuda.cupy.random.seed(fixed_seeds[0])
        cuda.numpy.random.seed(fixed_seeds[1])
                        

def cv_rotate(img, angle):
    # OpenCV doesn't work so well on Cedar
    # scikit-image's rotate function is almost 7x slower than OpenCV
    img = img.transpose(1, 2, 0) / 255.
    img = skimage_transform.rotate(img, angle, mode='edge')
    img = img.transpose(2, 0, 1) * 255.
    img = img.astype('f')
    return img

def transform_fast(inputs, cutout=None, flip=True, crop_size=None):
    """ Stripped down version of the transform function """
    img, label = inputs
    _, H, W = img.shape
    #img_orig = img
    img = img.copy()
    
    # Random flip
    if flip:
        img = transforms.random_flip(img, x_random=True)
        
    if crop_size is not None:
        h0 = np.random.randint(0, H-crop_size[0])
        w0 = np.random.randint(0, W-crop_size[1])
        img = img[:, h0:h0+crop_size[0], w0:w0+crop_size[1]]
    
    if cutout is not None:
        h0, w0 = np.random.randint(0, 32-cutout, size=(2,))
        img[:, h0:h0+cutout, w0:w0+cutout].fill(0.0)
    
    return img, label
    
def transform(inputs, mean=None, std=None, 
              random_angle=15., pca_sigma=255., expand_ratio=1.0,
              crop_size=(32, 32), cutout=None, flip=True, train=True):
    img, label = inputs
    img = img.copy()

    if train:
        # Random rotate
        if random_angle != 0:
            angle = np.random.uniform(-random_angle, random_angle)
            img = cv_rotate(img, angle)

        # Color augmentation
        if pca_sigma != 0:
            img = transforms.pca_lighting(img, pca_sigma)
        
        
    # Standardization
    if mean is not None:
        img -= mean[:, None, None]
        img /= std[:, None, None]

    if train:
        # Random flip
        if flip:
            img = transforms.random_flip(img, x_random=True)
            
        # Random expand
        if expand_ratio > 1:
            img = transforms.random_expand(img, max_ratio=expand_ratio)
            
        # Random crop
        if tuple(crop_size) != (32, 32) or expand_ratio > 1:
            img = transforms.random_crop(img, tuple(crop_size))
            
        # Cutout
        if cutout is not None:
            h0, w0 = np.random.randint(0, 32-cutout, size=(2,))
            img[:, h0:h0+cutout, w0:w0+cutout].fill(0.0)

    return img, label

def normalize_dataset(dataset, mean, std):
    if not isinstance(dataset, chainer.datasets.TupleDataset):
        raise ValueError('Expected TupleDataset')
    
    old_imgs, labels = dataset._datasets
    mean = mean.astype(old_imgs.dtype)
    std = std.astype(old_imgs.dtype)
    
    if not isinstance(old_imgs, np.ndarray):
        raise ValueError('Expected TupleDataset containing a tuple of numpy arrays')
    
    # Normalization
    imgs = (old_imgs - mean[None, :, None, None]) / std[None, :, None, None]
    
    return chainer.datasets.TupleDataset(imgs, labels)

def pad_dataset(dataset, pad=4):
    """ Pads the dataset using the reflect padding mode """
    imgs, labels = dataset._datasets
    imgs = np.pad(imgs, [(0, 0), (0,0), (pad, pad), (pad, pad)], mode='reflect')
    return chainer.datasets.TupleDataset(imgs, labels)