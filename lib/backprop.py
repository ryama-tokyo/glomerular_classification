import copy

import cv2
import numpy as np

import chainer
import chainer.functions as F

from lib.functions import GuidedReLU


class BaseBackprop(object):

    def __init__(self, model,n_out):
        self.model = model
        self.size = model.size
        self.xp = model.xp
        self.n_out=n_out

    def backward(self, x, label, layer):
        with chainer.using_config('train', False):
            acts = self.model.extract(x, layers=[layer, 'last'])

        one_hot = self.xp.zeros((3, self.n_out), dtype=np.float32)
        if label == -1:
            one_hot[:, acts['last'].data.argmax()] = 1
        else:
            one_hot[:, label] = 1

        self.model.cleargrads()
        prob=F.softmax(acts['last'])
        #print(prob)
        #print(prob.shape)
        #print(chainer.Variable(one_hot).shape)
        loss = F.sum(chainer.Variable(one_hot) * prob)
        loss = loss / 3 *1.0
        #loss = F.sum(chainer.Variable(one_hot) * acts['last'])
        loss.backward(retain_grad=True)

        return acts


class GradCAM(BaseBackprop):

    def __init__(self, model,n_out):
        super(GradCAM, self).__init__(model,n_out)

    def generate(self, x, label, layer):
        acts = self.backward(x, label, layer)
        feature=acts[layer]
        f_grad=acts[layer].grad
        triple_avg_f_grad=np.mean(f_grad,axis=(2,3))
        avg_f_grad=triple_avg_f_grad[0,:]
        gcam=np.zeros((feature.shape[2],feature.shape[3]))
        for k in range(avg_f_grad.shape[0]):
            gcam+=avg_f_grad[k]*feature[0,k,:,:]
        gcam=gcam.data
        #print(gcam>0)
        #print(gcam.max)
        gcam = (gcam > 0) * gcam / gcam.max()
        gcam = chainer.cuda.to_cpu(gcam * 255)
        gcam = cv2.resize(np.uint8(gcam), (self.size, self.size))

        return gcam


class GuidedBackprop(BaseBackprop):

    def __init__(self, model,n_out):
        super(GuidedBackprop, self).__init__(copy.deepcopy(model),n_out)
        for key, funcs in self.model.base.functions.items():
            if (key != 'fc6' and key!='prob'):
                for i in range(len(funcs)):
                    if funcs[i] is F.relu:
                        funcs[i] = GuidedReLU()

    def generate(self, x, label, layer):
        acts = self.backward(x, label, layer)
        #print(acts['input'].grad[0])
        gbp = chainer.cuda.to_cpu(acts['input'].grad[0])
        #print(gbp.shape)
        #print(gbp)
        gbp = gbp.transpose(1, 2, 0)

        return gbp
