# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_data(path):
    npz = np.load(path)
    x_train, y_train = npz['x_train'].astype(np.float32), npz['y_train'].astype(np.int32)
    x_test, y_test = npz['x_test'].astype(np.float32), npz['y_test'].astype(np.int32)
    return x_train, x_test, y_train, y_test

def make_learning_image(fig_path, mean_train_losses, mean_valid_losses):
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 7))
    k=max(max(mean_train_losses),max(mean_valid_losses))
    plt.ylim(0.0, k*1.2)
    plt.plot(mean_train_losses, "b", lw=1)
    plt.plot(mean_valid_losses, "r", lw=1)
    plt.title("")
    plt.ylabel("softmax cross entopy loss")
    plt.savefig(fig_path)
    plt.xlabel("epoch")
    plt.close()

def make_accuracy_image(fig_path, mean_train_accs, mean_valid_accs):
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 7))
    plt.ylim(0.0, 1)
    plt.plot(mean_train_accs, "b", lw=1)
    plt.plot(mean_valid_accs, "r", lw=1)
    plt.title("")
    plt.ylabel("accuracy")
    plt.savefig(fig_path)
    plt.xlabel("epoch")
    plt.close()

def make_f1score_image(fig_path, mean_train_accs, mean_test_accs,class_num):
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 7))
    plt.ylim(0.0, 1)
    plt.plot(mean_train_accs, "b", lw=1)
    plt.plot(mean_test_accs, "r", lw=1)
    plt.title("")
    plt.ylabel("class_"+str(class_num)+"_f1_score")
    plt.savefig(fig_path)
    plt.xlabel("epoch")
    plt.close()

if __name__ == '__main__':
    pass
    # path = "../99_data/200_MESA_RNDAUG_40K_CV1.npz"
    # npz = np.load(path)
    # print (npz.files)
    # x_train, t_train = npz['x_train'].astype(np.float32), npz['y_train'].astype(np.int32)
    # x_test, t_test = npz['x_test'].astype(np.float32), npz['y_test'].astype(np.int32)
    # print (x_train.shape, x_test.shape)
