
import os
import re
import pandas as pd
#%matplotlib inline
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
import yama_resnet50_revised as yama_net
import cupy
import argparse
from util import load_data, make_learning_image, make_accuracy_image
from cupy_augmentation import chw_cupy_random_rotate, chw_cupy_random_flip, cupy_augmentation

ordinal_dict={"1":"1st","2":"2nd","3":"3rd","4":"4th","5":"5th"}


class NNModel():
    def __init__(self, model_type,optm_type,prefix, dataset, gpu, flag_train, epoch, batchsize,alpha,
                 lr, dr, n_out, save_dir, save_interval, entropy_weight,patience_limit,resume_path=None):
        self.lr = lr
        self.alpha = alpha
        self.gpu = gpu
        self.prefix = prefix
        self.epoch = epoch
        self.save_dir = save_dir
        self.batchsize = batchsize
        self.flag_train = flag_train
        self.save_interval = save_interval
        self.entropy_weight = entropy_weight
        self.patience_limit = patience_limit
        self.n_out = n_out

        if model_type == 0: # Alex
            self.model = net.Alex(dr=dr, n_out=n_out,entropy_weight=entropy_weight)
        elif model_type == 1: # Google
            self.model = net.GoogLeNetBN(n_out=n_out,entropy_weight=entropy_weight)
        elif model_type == 2: # ResNet50
            self.model = net.Resnet50(n_out=n_out,entropy_weight=entropy_weight)
        elif model_type == 3: # ResNet152_transfer
            self.model = net.ResNet152_transfer(n_out=n_out,entropy_weight=entropy_weight)
            #self.model.predictor.base.disable_update()
        elif model_type == 4:
            self.model = yama_net.ResNet50Layers_transfer(n_out=n_out,entropy_weight=entropy_weight)
        else:
            print('wrong model type!')
            exit()


        # gpu check
        if gpu >= 0:
            #print('hoge')
            #chainer.backends.cuda.get_device_from_id(gpu).use()
            #print('hogehoge')
            chainer.cuda.get_device_from_id(gpu).use()
            #print(chainer.cuda.get_device_from_id(1))
            self.model.to_gpu()
            #self.model.to_gpu(gpu)

        # to train and valid (usually yes)
        if self.flag_train:
            if optm_type == 'adam': # 'ADAM'
                self.optimizer = chainer.optimizers.Adam(alpha=args.alpha)
                self.optimizer.setup(self.model)
            elif optm_type == 'momentum': # 'MomentumSGD'
                self.optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
                self.optimizer.setup(self.model)
            else:
                print('no optimizer set!')
                exit()

        # to resume from serialized model
        if resume_path is not None:
            try:
                chainer.serializers.load_npz(resume_path + '.model', self.model)
                chainer.serializers.load_npz(resume_path + '.state', self.optimizer)
                print('successfully resume model')
            except:
                print('WARN: cannot resume model')

        # prepare dataset
        self.train, self.valid = dataset[0], dataset[1]
        #print(len(dataset[0]))
        self.N_train, self.N_valid = len(train), len(valid)
        #print(self.N_train,self.N_valid)

        # data iterator
        '''
        self.train_iter = chainer.iterators.SerialIterator(self.train, self.batchsize, repeat=True, shuffle=True)
        self.valid_iter = chainer.iterators.SerialIterator(self.valid, self.batchsize, repeat=False, shuffle=False)
        '''
        self.train_iter = chainer.iterators.\
            MultithreadIterator(train, self.batchsize, repeat=True, shuffle=True)
        self.valid_iter = chainer.iterators.\
            MultithreadIterator(valid, self.batchsize, repeat=False, shuffle=False)
        '''
        self.train_iter = chainer.iterators.\
            MultiprocessIterator(train, self.batchsize, repeat=True, shuffle=True,
                                 n_processes=4, n_prefetch=8)
        self.valid_iter = chainer.iterators.\
            MultiprocessIterator(valid, self.batchsize, repeat=False, shuffle=False,
                                 n_processes=4, n_prefetch=8)
        '''


    def run(self):

        train_losses, valid_losses, train_accs, valid_accs = [], [], [], []
        #train_f1_scores, valid_f1_scores = [], []
        train_precision_scores, valid_precision_scores = [], []
        train_recall_scores, valid_recall_scores = [], []
        train_f1_scores, valid_f1_scores = [], []
        for j in range(self.n_out):
            train_precision_scores.append([])
            train_recall_scores.append([])
            train_f1_scores.append([])
            valid_precision_scores.append([])
            valid_recall_scores.append([])
            valid_f1_scores.append([])
        sum_train_loss, sum_train_accuracy = 0, 0
        all_train_t, all_valid_t=cupy.empty((0),cupy.int32),cupy.empty((0),cupy.int32)
        all_train_y, all_valid_y=cupy.empty((0,args.n_out),cupy.float32),cupy.empty((0,args.n_out),cupy.float32)
        best_valid_loss = np.inf
        best_valid_acc = np.inf

        #early stopping counter
        patience_counter=0


        while self.train_iter.epoch < self.epoch:

            # train phase
            batch = self.train_iter.next()
            if self.flag_train:

                # step by step update
                x_array, t_array = convert.concat_examples(batch, self.gpu)
                #print('x_array',x_array.shape,type(x_array))
                #print('t_array',t_array.shape,type(t_array))
                all_train_t=cupy.hstack([all_train_t,t_array])

                x, t = chainer.Variable(x_array), chainer.Variable(t_array)
                x=cupy_augmentation(x)#added at 2019/09/10

                self.model.cleargrads()
                y, loss, accuracy ,_ = self.model(x, t)
                #y_for_f1=cupy.argmax(y.data,axis=1)
                all_train_y=cupy.vstack([all_train_y,y.data])
                #print(all_train_y.shape

                loss.backward()
                self.optimizer.update()

                sum_train_loss += float(loss.data) * len(t.data)
                sum_train_accuracy += float(accuracy.data) * len(t.data)
            #train_f1_score=F.classification_s

            # valid phase
            if self.train_iter.is_new_epoch:

                # Return objects Loss
                mean_train_loss = sum_train_loss / self.N_train
                train_losses.append(mean_train_loss)

                # Return objects Acc
                mean_train_acc = sum_train_accuracy / self.N_train
                train_accs.append(mean_train_acc)

                # Return objects f1_score
                #train_f1_score=F.classification_summary(all_train_y,all_train_t)[2][1]
                #print(train_f1_score)
                #train_f1_scores.append(train_f1_score)
                for j in range(self.n_out):
                    train_precision_score=F.classification_summary(all_train_y,all_train_t)[0][j]
                    train_precision_scores[j].append(train_precision_score)
                    train_recall_score=F.classification_summary(all_train_y,all_train_t)[1][j]
                    train_recall_scores[j].append(train_recall_score)
                    train_f1_score=F.classification_summary(all_train_y,all_train_t)[2][j]
                    train_f1_scores[j].append(train_f1_score)


                sum_valid_accuracy, sum_valid_loss = 0, 0
                all_train_t, all_valid_t=cupy.empty((0),cupy.int32),cupy.empty((0),cupy.int32)
                all_train_y, all_valid_y=cupy.empty((0,args.n_out),cupy.float32),cupy.empty((0,args.n_out),cupy.float32)

                for batch in self.valid_iter:
                    x_array, t_array = convert.concat_examples(batch, self.gpu)
                    all_valid_t=cupy.hstack([all_valid_t,t_array])
                    #t_for_f1=np.argmax(t_array,axis=1)
                    x, t = chainer.Variable(x_array), chainer.Variable(t_array)

                    with chainer.using_config('train', False), chainer.no_backprop_mode():
                        y, loss, accuracy,f1_score = self.model(x, t)
                        #y_for_f1=cupy.argmax(y.data,axis=1)
                        #print('y_for_f1',y_for_f1.shape)

                    sum_valid_loss += float(loss.data) * len(t.data)
                    sum_valid_accuracy += float(accuracy.data) * len(t.data)
                    all_valid_y=cupy.vstack([all_valid_y,y.data])

                # Return objects Loss
                mean_valid_loss = sum_valid_loss / self.N_valid
                valid_losses.append(mean_valid_loss)

                # Return objects valid
                mean_valid_acc = sum_valid_accuracy / self.N_valid
                valid_accs.append(mean_valid_acc)

                # Return objects f1_score
                #print(all_valid_y.dtype,all_valid_t.dtype)
                #print(all_valid_y.shape,all_valid_t.shape)
                #print(np.max(all_valid_t))
                #print(F.classification_summary(all_valid_y,all_valid_t))
                for j in range(self.n_out):
                    valid_precision_score=F.classification_summary(all_valid_y,all_valid_t)[0][j]
                    valid_precision_scores[j].append(valid_precision_score)
                    valid_recall_score=F.classification_summary(all_valid_y,all_valid_t)[1][j]
                    valid_recall_scores[j].append(valid_recall_score)
                    valid_f1_score=F.classification_summary(all_valid_y,all_valid_t)[2][j]
                    valid_f1_scores[j].append(valid_f1_score)

                self.valid_iter.reset()

                if mean_valid_loss < best_valid_loss:
                    # update best
                    best_valid_loss = mean_valid_loss
                    best_valid_acc = mean_valid_acc
                    #print(train_f1_score.data)
                    print("e %d/%d, train_loss %f, valid_loss(Best) %f, train_accuracy %f, valid_accuracy %f ,train_f1 %f , valid_f1 %f" % (
                        self.train_iter.epoch, args.epoch, mean_train_loss, best_valid_loss,
                        mean_train_acc, mean_valid_acc,train_f1_score.data,valid_f1_score.data))
                    save_flag = 1
                    patience_counter = 0#Important! reset the patience_counter

                else:
                    patience_counter = patience_counter+1
                    print('patience_counter is accumulated, counter is '+ str(patience_counter))
                    save_flag = 0
                    print("e %d/%d, train_loss %f, valid_loss %f, train_accuracy %f, valid_accuracy %f ,train_f1 %f, valid_f1 %f" % (
                        self.train_iter.epoch, args.epoch, mean_train_loss, mean_valid_loss,
                        mean_train_acc, mean_valid_acc, train_f1_score.data, valid_f1_score.data))

                sum_train_loss, sum_train_accuracy = 0, 0
                all_train_t, all_valid_t=cupy.empty((0),cupy.int32),cupy.empty((0),cupy.int32)
                all_train_y, all_valid_y=cupy.empty((0,args.n_out),cupy.float32),cupy.empty((0,args.n_out),cupy.float32)

                if self.save_interval > 0 :
                    if self.train_iter.epoch % self.save_interval == 0 or self.train_iter.epoch == self.epoch or save_flag == 1:
                        try:
                            chainer.serializers.save_npz(save_dir + '/' + self.prefix + "_e" + str(self.train_iter.epoch) + '.model', self.model)
                            chainer.serializers.save_npz(save_dir + '/' + self.prefix + "_e" + str(self.train_iter.epoch) + '.state', self.optimizer)
                            print('Successfully saved model')
                        except:
                            print('WARN: saving model ignored')

            # early stopping
            if patience_counter >= patience_limit:
                break
        return train_losses, valid_losses, train_accs, valid_accs, best_valid_loss ,train_f1_scores,valid_f1_scores,train_precision_scores,valid_precision_scores,train_recall_scores,valid_recall_scores,best_valid_acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Chainer v4.0.0')
    parser.add_argument('--train', '-t', type=int, default=1, help='If negative, skip training')
    parser.add_argument('--n_fold', '-nf', type=int, default=5, help='The number of K-fold')
    parser.add_argument('--batchsize', '-b', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=3000, help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='Main GPU ID')
    parser.add_argument('--save_interval', '-s', type=int, default=500, help='interval for saving model')
    parser.add_argument('--lr', '-lr', type=float, default=0.01, help='Learning Rate')
    parser.add_argument('--alpha', '-alpha', type=float, default=0.001, help='Alpha of Adam')
    parser.add_argument('--dr', '-dr', type=float, default=0.0, help='Dropout Rate')
    parser.add_argument('--n_out', '-no', type=int, default=4, help='Number of output class')
    parser.add_argument('--model_type', '-modeltype', type=int, default=0, help='0:Alex, 1:GoogLe, 2:ResNet50')
    parser.add_argument('--optm_type', '-optmtype', type=str, default='no optm is set', help='adam:ADAM, momentum:momentumSGD')
    #parser.add_argument('--save_dir', '-save_dir', type=str, default='empty_folder', help='save_dir')
    parser.add_argument('--remark', '-remark', type=str, default='hoge', help='remarks:e.g meshyper')
    parser.add_argument('--weighted_loss', '-weighted_loss', type=int, default=1, help='use weighted cross entropy if 1')
    #parser.add_argument('--entropy_weight', '-entropy_weight', type=float, default=1, help='entropy_weight',nargs='*')
    parser.add_argument('--patience_limit', '-patience_limit', type=int, default=100, help='early stop counter,how many times new valid loss is over best valid loss')
    args = parser.parse_args()

    # remark name
    npy_filename = args.remark
    #print(npy_filename)
    n_fold = args.n_fold

    # path to data
    data_dir = './dataset/folded_npy/'

    npy_filename = args.remark
    patience_limit= args.patience_limit

    # check saving directory
    save_dir=os.getcwd() + '/result/' + npy_filename + '/resnet50_transfer/'
    if not os.path.exists(save_dir):
        print("no save dir", save_dir)
        exit()


    # check resume model if use
    resume_path = None
    if resume_path is not None:
        if not os.path.exists(resume_path):
            print("no resume model", resume_path)
            exit()

    # misc
    flag_train = False if args.train < 0 else True
    n_epoch = args.epoch if flag_train == True else 1

    # print status
    print('GPU: {}'.format(args.gpu))
    print('# epoch: {}'.format(args.epoch))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# dropout: {}'.format(args.dr))
    print('# learning rate: {}'.format(args.lr))

    # for figure
    all_train_losses, all_valid_losses = [], []
    all_train_accs, all_valid_accs = [], []
    all_best_valid_losses = []
    all_train_precision_scores,all_valid_precision_scores= [], []
    all_train_recall_scores,all_valid_recall_scores= [], []
    all_train_f1_scores,all_valid_f1_scores= [], []
    all_best_valid_accs = []

    entropy_weight=[]
    weight_sheet=pd.read_csv('/workspace/mountNAS/99_WS/10_experimental_data/ryama_organized/20191126_folding_remake/weight_of_remark_new.csv',index_col=0)
    weight_sheet=weight_sheet.rename(columns={'0': 'remark'})
    location = (np.where(weight_sheet['remark'] == npy_filename)[0])
    print(weight_sheet.iloc[location+0,:])
    print(weight_sheet.iloc[location+1,:])
    print(weight_sheet.iloc[location+2,:])
    print(weight_sheet.iloc[location+3,:])
    if args.weighted_loss == 1:
        for j in range(args.n_out):
            weight_jth = weight_sheet.iloc[location+3,j]
            weight_jth = float(weight_jth)
            entropy_weight.append(weight_jth)
    else:
        for j in range(args.n_out):
            weight_jth = 1
            weight_jth = float(weight_jth)
            entropy_weight.append(weight_jth)

    # time watch
    print("start:", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    #print(args.n_out)

    # start eval
    for ncv in range(0, n_fold):
        print(str(ncv+1)+'folding_calculation_start')
        train_path = data_dir + 'train_' + ordinal_dict[str(ncv+1)]+'/'
        valid_path = data_dir + 'valid_' + ordinal_dict[str(ncv+1)]+'/'
        print("load train data from " + train_path)
        x_train=np.load(train_path + 'train_' + ordinal_dict[str(ncv+1)] + '_images.npy')
        y_train=np.load(train_path + 'train_' + ordinal_dict[str(ncv+1)] + '_' + npy_filename + '_label.npy')

        print("load valid data from " + valid_path)
        x_valid=np.load(valid_path + 'valid_' + ordinal_dict[str(ncv+1)] + '_images.npy')
        y_valid=np.load(valid_path + 'valid_' + ordinal_dict[str(ncv+1)] + '_' + npy_filename + '_label.npy')

        train = chainer.datasets.tuple_dataset.TupleDataset(x_train, y_train)
        valid = chainer.datasets.tuple_dataset.TupleDataset(x_valid, y_valid)

        prefix = str(args.model_type) + "_" + str(ncv+1) + 'thFold_'+npy_filename + "_b" + str(args.batchsize) + "_optm_"+str(args.optm_type)+"_alpha" + str(args.alpha) + "_lr" + str(args.lr)
        my_network = NNModel(model_type=args.model_type, optm_type=args.optm_type,prefix=prefix, dataset=[train, valid], gpu=args.gpu,
                          flag_train=flag_train, epoch=args.epoch, batchsize=args.batchsize,alpha=args.alpha,
                          lr=args.lr, dr=args.dr, n_out=args.n_out, save_dir=save_dir,entropy_weight=entropy_weight,
                          save_interval=args.save_interval, resume_path=resume_path,patience_limit=patience_limit)

        train_losses, valid_losses, train_accs, valid_accs, best_valid_loss ,train_f1_scores,valid_f1_scores,train_precision_scores,valid_precision_scores,train_recall_scores,valid_recall_scores,best_valid_acc= my_network.run()

        all_train_losses.append(train_losses)
        all_valid_losses.append(valid_losses)
        all_train_accs.append(train_accs)
        all_valid_accs.append(valid_accs)
        all_best_valid_losses.append(best_valid_loss)
        all_train_precision_scores.append(train_precision_scores)
        all_valid_precision_scores.append(valid_precision_scores)
        all_train_recall_scores.append(train_recall_scores)
        all_valid_recall_scores.append(valid_recall_scores)
        all_train_f1_scores.append(train_f1_scores)
        all_valid_f1_scores.append(valid_f1_scores)
        all_best_valid_accs.append(best_valid_acc)

    print ("making figure")

    # early stoppingを実装した場合、fold毎に終了epochが違うため、epoch数を揃える必要がある。
    # for rangeループの中に含まれるものはarray集合のリストであり、長さが不揃いになっているためエポック数を揃える。
    #print(all_train_losses.dtype,all_train_losses.shape,'all_train')
    #print(all_valid_losses.dtype,all_valid_losses.shape,'all_valid')
    if patience_limit >= 0:
        print('early stopping, so ajusting the number of epoch for each folding.')
        n_cv1 = len(all_valid_losses[0])
        n_cv2 = len(all_valid_losses[1])
        n_cv3 = len(all_valid_losses[2])
        n_cv4 = len(all_valid_losses[3])
        #n_cv5 = len(all_valid_losses[4])
        #min_index = min(n_cv1,n_cv2)
        min_index = min(n_cv1,n_cv2,n_cv3,n_cv4)
        for j in range(n_fold):
            all_train_losses[j] = (all_train_losses[j])[0:min_index]
            all_valid_losses[j] = (all_valid_losses[j])[0:min_index]
            all_train_accs[j] = (all_train_accs[j])[0:min_index]
            all_valid_accs[j] = (all_valid_accs[j])[0:min_index]

    # figure
    mean_best_valid_losses = np.array(all_best_valid_losses).mean(axis=0)
    mean_best_valid_accs  = np.array(all_best_valid_accs).mean(axis=0)
    mean_train_losses, mean_valid_losses = np.array(all_train_losses).mean(axis=0), np.array(all_valid_losses).mean(axis=0)
    mean_train_accs, mean_valid_accs = np.array(all_train_accs).mean(axis=0), np.array(all_valid_accs).mean(axis=0)

    # loss
    filename_prefix = str(args.model_type) + "_remark_" + npy_filename + "_b" + str(args.batchsize) + "_alpha" + str(args.alpha)
    fig_path = os.getcwd() + '/result/' + npy_filename + '/' + filename_prefix + "_best_" + str(mean_best_valid_losses) + "_loss.png"
    make_learning_image(fig_path=fig_path, mean_train_losses=mean_train_losses, mean_valid_losses=mean_valid_losses)

    # acc
    fig_path = os.getcwd() + '/result/' + npy_filename + '/' + filename_prefix + "_best_" + str(mean_best_valid_accs) + "_acc.png"
    make_accuracy_image(fig_path=fig_path, mean_train_accs=mean_train_accs, mean_valid_accs=mean_valid_accs)


    if args.save_interval > 0:
        try:
            np_save_point = os.getcwd() + '/result/' + npy_filename + '/'
            np.save(np_save_point+'all_train_f1_scores',all_train_f1_scores)
            np.save(np_save_point+'all_train_precision_scores',all_train_precision_scores)
            np.save(np_save_point+'all_train_recall_scores',all_train_recall_scores)
            np.save(np_save_point+'all_valid_f1_scores',all_valid_f1_scores)
            np.save(np_save_point+'all_valid_precision_scores',all_valid_precision_scores)
            np.save(np_save_point+'all_valid_recall_scores',all_valid_recall_scores)
        except:
            pass

    # time watch
    print("finish:", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
