import chainer
import copy
import cupy
import random

def chw_cupy_random_flip(chw_imageset,y_random=False,x_random=False,return_param=False,copy=False):
    if len(chw_imageset.shape) == 3:
        chw_imageset=cupy.expand_dims(chw_imageset,axis=0)
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        chw_imageset = chw_imageset[:,:, ::-1, :]
    if x_flip:
        chw_imageset = chw_imageset[:,:, :, ::-1]

    if copy:
        chw_imageset = chw_imageset.copy()

    if return_param:
        return chw_imageset, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return chw_imageset

def chw_cupy_random_rotate(chw_imageset,return_param=False):
    if len(chw_imageset.shape) == 3:
        chw_imageset=cupy.expand_dims(chw_imageset,axis=0)
    k=cupy.random.randint(4)
    pil_imageset=cupy.transpose(chw_imageset,axes=(0,2,3,1))
    pil_rotated_imageset = cupy.rot90(pil_imageset,k,axes=(1,2))#pil_imgはn,x,y,colorになっている。axis=(1,2)で指定
    chw_rotated_imageset = cupy.transpose(pil_rotated_imageset,axes=(0,3,1,2))
    if return_param:
        return chw_rotated_imageset,{'k':k}
    else:
        return chw_rotated_imageset

def cupy_augmentation(chw_imageset):
    flipped_imageset=chw_cupy_random_flip(chw_imageset,y_random=True,x_random=False,return_param=False)
    flipped_rotated_imageset=chw_cupy_random_rotate(flipped_imageset,return_param=False)
    return flipped_rotated_imageset
