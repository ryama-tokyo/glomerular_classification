import os
#import seaborn as sns
#import matplotlib.pyplot as plt
import re
from datetime import datetime
from PIL import Image
import numpy as np
import sys
import shutil
from distutils import dir_util
import re
import glob
import chainer
import chainer.links as L
import chainer.functions as F
import chainer.cuda as cuda
from chainer.dataset import convert
import cupy
import argparse
import pandas as pd
#import matplotlib.pyplot as plt

import os
import argparse

import chainer
import numpy as np
import cv2
#import matplotlib.pyplot as plt
#from VGGNet import VGGNet
from chainer import cuda
from chainer import serializers
from chainer import Variable
from chainer import initializers
import chainer.functions as F
import chainer.links as L
import h5py
from yama_resnet50_revised import ResNet50Layers_transfer
from yama_resnet50_revised import ResNet50Layers
from PIL import Image
import cupy
import copy
from lib.functions import GuidedReLU
from lib import backprop
from chainer.links.connection.linear import Linear


making_image_directory_list=['collapsingobsolete_exist_new','fibrouscrescent']
#collapsingobsolete_global_new

making_image_directory_dict={'all_scl_exist_new':2,'all_scl_global_new':2,\
                       'collapsingobsolete_exist_new':2,'fibrouscrescent':2,\
                       'fibrouscrescent':2,'fc_and_c':2,\
                       'matrixincrease_new':2,'collapsingobsolete_global_new':2,\
                       'mesangiolysis_new':2,'polarvasculosis_new':2,\
                       'extracellular_exist':3,'extracellular_global':3}
order_dict={'1':'1st','2':'2nd','3':'3rd','4':'4th','5':'5th'}
def strdata_to_int(text):
    tmp=re.sub("[()]","",text)
    tmp=re.sub(r'[a-z]+', "", tmp)
    tmp=re.sub(r'[A-Z]+', "", tmp)
    tmp=re.sub('_', "", tmp)
    tmp=tmp.split('/')
    int_data=int(tmp[0])
    return int_data
def strdata_to_float(text):
    tmp=re.sub("[()]","",text)
    tmp=re.sub('f1','f',tmp)
    tmp=re.sub(r'[a-z]+', "", tmp)
    tmp=re.sub(r'[A-Z]+', "", tmp)
    tmp=re.sub('_', "", tmp)
    tmp=re.sub('/', "", tmp)
    float_data=float(tmp)
    return float_data

def my_round(val, digit=0):
    p = 10 ** digit
    return (val * p * 2 + 1) // 2 / p

def return_epoch_of_min_valid_loss(abs_txt_path,n_of_fold):
    # returns epoch of min_valid_loss
    data_txt=pd.read_table(abs_txt_path,names=('epoch', 'train_loss', 'valid_loss', 'train_accuracy','valid_accuracy','train_f1','valid_f1'),delimiter=',')
    valid_losses=np.empty((n_of_fold,2000))
    valid_losses[:,:]=np.inf
    min_count_array=np.zeros(n_of_fold)
    epoch = 1#epoch starts with 1 ,numpy count starts with 0,notice!
    fold_count = -1
    for l in range(len(data_txt)):
        if 'folding_calculation' in data_txt.iloc[l,0]:
            fold_count = fold_count + 1
            epoch = 1
        elif data_txt.iloc[l,0].startswith('e') and 'early' not in data_txt.iloc[l,0]:
            valid_loss = strdata_to_float(data_txt.iloc[l,2])
            valid_losses[fold_count,epoch-1] = valid_loss
            epoch = epoch + 1
    for j in range(n_of_fold):
        min_count = np.argmin(valid_losses[j,:])
        min_count_array[j] = min_count
    min_count_array=min_count_array+1# epoch = numpy_index +1
    return valid_losses,min_count_array
weight_sheet=pd.read_csv('./dataset/weight_of_remark_new.csv')
weight_sheet=weight_sheet.rename(columns={'0': 'remark'})

class ResNet50_grad(chainer.Chain):

    def __init__(self, n_out,entropy_weight):
        super(ResNet50_grad, self).__init__()
        with self.init_scope():
            self.base = ResNet50Layers(pretrained_model='auto')
            self.fc7  = Linear(None,n_out)
        #self.base=base
        self.entropy_weight=entropy_weight
        self.n_out=n_out
        self.size=224

    def __call__(self, x ,layers=None, **kwags):
        #h = x
        '''
        if len(x.shape) == 3:
            x=x.reshape(1,3,224,224)
        '''
        h=x
        activations = {'input': x}
        target_layers = set(layers)
        #print(target_layers)
        functions_ordered_dict=self.base.functions()
        for key, funcs in functions_ordered_dict.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                #print(h.shape,func)
                h = func(h)
                #print(h.shape)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        #print(h.shape,self.fc,'hogehoge')
        #print(h.shape)
        h=self.fc7(h)
        #print(h.shape)
        #print(h.shape)
        activations['last'] = h
        return activations

    def extract(self, x, layers=['fc7']):
        x = chainer.Variable(self.xp.asarray(x))
        #print(x.shape)
        return self(x,layers=layers)

for k in range(len(making_image_directory_list)):
    #csv_sheet = pd.DataFrame(index=range(12000), columns=['filname','true_class','class0_prob','class1_prob','class2_prob','class3_prob'])
    remark=making_image_directory_list[k]
    print(remark)

    n_class = making_image_directory_dict[remark]

    result_dir= './experiment/result/'+remark+'/resnet50_transfer/'
    result_txt= result_dir +'result.txt'
    min_epoch_list = return_epoch_of_min_valid_loss(result_txt,4)[1]
    result_txt
    test_dir='./dataset/folded_npy/test/'
    #test_dir='/workspace/mountNAS/99_WS/10_experimental_data/ryama_organized/20191204_commondata_make/common100_data/'

    grad_save_dir = './experiment/result/'

    entropy_weight=[]
    location = (np.where(weight_sheet['remark'] == remark)[0])
    for j in range(n_class):
        weight_jth = 1
        weight_jth = float(weight_jth)
        entropy_weight.append(weight_jth)

    #n_row=0
    for j in range(1):
        n_fold = j + 1
        print(n_fold,'th_fold start')
        data_dir= test_dir + 'test_images.npy'
        label_dir= test_dir + 'test_' + remark + '_label.npy'
        filename_dir= test_dir + 'test_filenames.npy'
        common_x=np.load(data_dir)
        common_y=np.load(label_dir)
        common_z=np.load(filename_dir)

        #print(test_iter)
        #print(N_test)

        n_class = making_image_directory_dict[remark]
        min_epoch_of_jth_fold = int(min_epoch_list[j])

        best_res50_trans_path=result_dir+'4_' + str(n_fold) +'thFold_'+remark+'_b50_optm_adam_alpha1e-07_lr0.01_e'+str(min_epoch_of_jth_fold)+'.model'
        # model load
        model_res50_trans=ResNet50_grad(n_out=n_class,entropy_weight=entropy_weight)#entropy_weightは予測に影響を及ぼさない
        serializers.load_npz(best_res50_trans_path,model_res50_trans)

        #gpu_device = 2
        #cuda.get_device(gpu_device).use()
        #print(common_z)
        for m in range(common_z.shape[0]):
            if m % 100 == 0:
                print(m)
            filename=common_z[m]
            filename_code=filename
            filename_code=filename_code.replace('.png','')
            sample_data=common_x[m,:,:,:]
            sample_data=sample_data[np.newaxis,:,:,:]
            target_data=np.vstack((sample_data,sample_data,sample_data))
            with chainer.using_config('train', False):
                act=model_res50_trans.extract(target_data,layers=['last','res5'])
            grad_cam = backprop.GradCAM(model_res50_trans,n_out=n_class)
            gcam = grad_cam.generate(target_data, 1, 'res5')
            heatmap = cv2.applyColorMap(gcam, cv2.COLORMAP_JET)

            filedata=common_x[m,:,:,:]
            filedata=filedata*255/filedata.max()
            r=Image.fromarray(np.uint8(filedata[0,:,:]))
            g=Image.fromarray(np.uint8(filedata[1,:,:]))
            b=Image.fromarray(np.uint8(filedata[2,:,:]))
            pillowdata=Image.merge("RGB",(r,g,b))
            pillowname=grad_save_dir + remark + '/'+filename
            pillowdata.save(pillowname)

            src=cv2.imread(pillowname,1)

            gcam = np.float32(src) + np.float32(heatmap)
            gcam = 255 * gcam / gcam.max()
            gcam_filename = grad_save_dir + remark + '/' + filename_code +'_'+str(n_fold)+ 'th_fold'+'.png'
            cv2.imwrite(gcam_filename, gcam)


        del model_res50_trans
