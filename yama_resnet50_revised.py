from __future__ import print_function
import collections
import os

import numpy
import cupy

try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e

import chainer
import chainer.functions as F
from chainer.dataset.convert import concat_examples
from chainer.dataset import download
from chainer import function
from chainer.functions.activation.relu import relu
from chainer.functions.activation.softmax import softmax
from chainer.functions.array.reshape import reshape
from chainer.functions.math.sum import sum
from chainer.functions.pooling.average_pooling_2d import average_pooling_2d
from chainer.functions.pooling.max_pooling_2d import max_pooling_2d
from chainer.initializers import constant
from chainer.initializers import normal
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.linear import Linear
from chainer.links.normalization.batch_normalization import BatchNormalization
from chainer.serializers import npz
from chainer.utils import argument
from chainer.utils import imgproc
from chainer.variable import Variable

class ResNetLayers(link.Chain):

    """A pre-trained CNN model provided by MSRA.
    When you specify the path of the pre-trained chainer model serialized as
    a ``.npz`` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    This model would be useful when you want to extract a semantic feature
    vector per image, or fine-tune the model on a different dataset.
    Note that unlike ``VGG16Layers``, it does not automatically download a
    pre-trained caffemodel. This caffemodel can be downloaded at
    `GitHub <https://github.com/KaimingHe/deep-residual-networks>`_.
    If you want to manually convert the pre-trained caffemodel to a chainer
    model that can be specified in the constructor,
    please use ``convert_caffemodel_to_npz`` classmethod instead.
    See: K. He et. al., `Deep Residual Learning for Image Recognition
    <https://arxiv.org/abs/1512.03385>`_
    Args:
        pretrained_model (str): the destination of the pre-trained
            chainer model serialized as a ``.npz`` file.
            If this argument is specified as ``auto``,
            it automatically loads and converts the caffemodel from
            ``$CHAINER_DATASET_ROOT/pfnet/chainer/models/ResNet-{n-layers}-model.caffemodel``,
            where ``$CHAINER_DATASET_ROOT`` is set as
            ``$HOME/.chainer/dataset`` unless you specify another value
            by modifying the environment variable and {n_layers} is replaced
            with the specified number of layers given as the first argment to
            this costructor. Note that in this case the converted chainer
            model is stored on the same directory and automatically used from
            the next time.
            If this argument is specified as ``None``, all the parameters
            are not initialized by the pre-trained model, but the default
            initializer used in the original paper, i.e.,
            ``chainer.initializers.HeNormal(scale=1.0)``.
        n_layers (int): The number of layers of this model. It should be either
            50, 101, or 152.
    Attributes:
        ~ResNetLayers.available_layers (list of str): The list of available
            layer names used by ``__call__`` and ``extract`` methods.
    """

    #def __init__(self,n_out,entropy_weight,n_layers,pretrained_model):
    def __init__(self,pretrained_model,n_layers):
        super(ResNetLayers, self).__init__()

        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            kwargs = {'initialW': constant.Zero()}
        else:
            # employ default initializers used in the original paper
            kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        if n_layers == 50:
            block = [3, 4, 6, 3]
        elif n_layers == 101:
            block = [3, 4, 23, 3]
        elif n_layers == 152:
            block = [3, 8, 36, 3]
        else:
            raise ValueError('The n_layers argument should be either 50, 101,'
                             ' or 152, but {} was given.'.format(n_layers))

        with self.init_scope():
            self.conv1 = Convolution2D(3, 64, 7, 2, 3, **kwargs)
            self.bn1 = BatchNormalization(64)
            self.res2 = BuildingBlock(block[0], 64, 64, 256, 1, **kwargs)
            self.res3 = BuildingBlock(block[1], 256, 128, 512, 2, **kwargs)
            self.res4 = BuildingBlock(block[2], 512, 256, 1024, 2, **kwargs)
            self.res5 = BuildingBlock(block[3], 1024, 512, 2048, 2, **kwargs)
            self.fc6 = Linear(2048, 1000)
            #self.fc7 = Linear(1000, n_out) adding in consutruct of res50layer_transfer

        if pretrained_model and pretrained_model.endswith('.caffemodel'):
            _retrieve(n_layers, 'ResNet-{}-model.npz'.format(n_layers),
                      pretrained_model, self)
        elif pretrained_model:
            npz.load_npz(pretrained_model, self)
    '''
    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, relu]),
            ('pool1', [lambda x: max_pooling_2d(x, ksize=3, stride=2)]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
            ('res5', [self.res5]),
            ('pool5', [_global_average_pooling_2d]),
            ('fc6', [self.fc6]),
            ('fc7', [self.fc7]),
            ('prob', [softmax_cross_entropy]),
        ])
    '''
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, relu]),
            ('pool1', [lambda x: max_pooling_2d(x, ksize=3, stride=2)]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
            ('res5', [self.res5]),
            ('pool5', [_global_average_pooling_2d]),
            ('fc6', [self.fc6]),
        ])

    @property
    def available_layers(self):
        return list(self.functions.keys())

    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz, n_layers=50):
        """Converts a pre-trained caffemodel to a chainer model.
        Args:
            path_caffemodel (str): Path of the pre-trained caffemodel.
            path_npz (str): Path of the converted chainer model.
        """

        # As CaffeFunction uses shortcut symbols,
        # we import CaffeFunction here.
        from chainer.links.caffe.caffe_function import CaffeFunction
        caffemodel = CaffeFunction(path_caffemodel)
        chainermodel = cls(pretrained_model=None, n_layers=n_layers)
        if n_layers == 50:
            _transfer_resnet50(caffemodel, chainermodel)
        elif n_layers == 101:
            _transfer_resnet101(caffemodel, chainermodel)
        elif n_layers == 152:
            _transfer_resnet152(caffemodel, chainermodel)
        else:
            raise ValueError('The n_layers argument should be either 50, 101,'
                             ' or 152, but {} was given.'.format(n_layers))
        npz.save_npz(path_npz, chainermodel, compression=False)

    '''
    def __call__(self, x, t):
        h = x
        for key, funcs in self.functions.items():
            h = func(h)#return score of fc7
        cw = cupy.array([self.entropy_weight[0]]).astype(cupy.float32)
        if len(self.entropy_weight) > 1:
            for k in range(len(self.entropy_weight)-1):
                tmp_weight=self.entropy_weight[k+1]
                tmp=cupy.array([tmp_weight]).astype(cupy.float32)
                cw = cupy.concatenate([cw,tmp])
        loss, accuracy = F.softmax_cross_entropy(h, t,class_weight=cw), F.accuracy(h, t)
        f1_score=F.f1_score(h,t)
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        return h, loss, accuracy,f1_score

    def calc_activation(self, x, layers=None, **kwargs):
        """calc_activation(self, x, layers=['fc7'])
        Computes all the feature maps specified by ``layers``.
        .. warning::
           ``test`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', train)``.
           See :func:`chainer.using_config`.
        Args:
            x (~chainer.Variable): Input variable. It should be prepared by
                ``prepare`` function.
            layers (list of str): The list of layer names you want to extract.
        Returns:
            Dictionary of ~chainer.Variable: A directory in which
            the key contains the layer name and the value contains
            the corresponding feature map variable.
        """

        if layers is None:
            layers = ['fc7']

        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

        h = x
        activations = {}
        target_layers = set(layers)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        return activations

    def extract(self, images, layers=None, size=(224, 224), **kwargs):
        """extract(self, images, layers=['pool5'], size=(224, 224))
        Extracts all the feature maps of given images.
        The difference of directly executing ``__call__`` is that
        it directly accepts images as an input and automatically
        transforms them to a proper variable. That is,
        it is also interpreted as a shortcut method that implicitly calls
        ``prepare`` and ``__call__`` functions.
        .. warning::
           ``test`` and ``volatile`` arguments are not supported anymore since
           v2.
           Instead, use ``chainer.using_config('train', train)`` and
           ``chainer.using_config('enable_backprop', not volatile)``
           respectively.
           See :func:`chainer.using_config`.
        Args:
            images (iterable of PIL.Image or numpy.ndarray): Input images.
            layers (list of str): The list of layer names you want to extract.
            size (pair of ints): The resolution of resized images used as
                an input of CNN. All the given images are not resized
                if this argument is ``None``, but the resolutions of
                all the images should be the same.
        Returns:
            Dictionary of ~chainer.Variable: A directory in which
            the key contains the layer name and the value contains
            the corresponding feature map variable.
        """

        if layers is None:
            layers = ['pool5']

        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
            'Use chainer.using_config',
            volatile='volatile argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

        #x = concat_examples([prepare(img, size=size) for img in images])
        x = images
        x = Variable(self.xp.asarray(x))
        #return self(x, layers=layers)
        return self.calc_activation(x,layers=layers)

    #def predict(self, images, oversample=True):
    def predict(self, images, oversample=False):
        """Computes all the probabilities of given images.
        Args:
            images (iterable of PIL.Image or numpy.ndarray): Input images.
                When you specify a color image as a :class:`numpy.ndarray`,
                make sure that color order is RGB.
            oversample (bool): If ``True``, it averages results across
                center, corners, and mirrors. Otherwise, it uses only the
                center.
        Returns:
            ~chainer.Variable: Output that contains the class probabilities
            of given images.
        """
        #x = concat_examples([prepare(img, size=(256, 256)) for img in images])
        #x = concat_examples([prepare(img, size=(224, 224)) for img in images])
        # Use no_backprop_mode to reduce memory consumption
        x = images
        with function.no_backprop_mode(), chainer.using_config('train', False):
            x = Variable(self.xp.asarray(x))
            y = self.extract(x, layers=['fc7'])['fc7']
            y = softmax(y)#probability calc
        return y
    '''


class ResNet50Layers(ResNetLayers):

    """A pre-trained CNN model with 50 layers provided by MSRA.
    When you specify the path of the pre-trained chainer model serialized as
    a ``.npz`` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    This model would be useful when you want to extract a semantic feature
    vector per image, or fine-tune the model on a different dataset.
    Note that unlike ``VGG16Layers``, it does not automatically download a
    pre-trained caffemodel. This caffemodel can be downloaded at
    `GitHub <https://github.com/KaimingHe/deep-residual-networks>`_.
    If you want to manually convert the pre-trained caffemodel to a chainer
    model that can be specified in the constructor,
    please use ``convert_caffemodel_to_npz`` classmethod instead.
    ResNet50 has 25,557,096 trainable parameters, and it's 58% and 43% fewer
    than ResNet101 and ResNet152, respectively. On the other hand, the top-5
    classification accuracy on ImageNet dataset drops only 0.7% and 1.1% from
    ResNet101 and ResNet152, respectively. Therefore, ResNet50 may have the
    best balance between the accuracy and the model size. It would be basically
    just enough for many cases, but some advanced models for object detection
    or semantic segmentation use deeper ones as their building blocks, so these
    deeper ResNets are here for making reproduction work easier.
    See: K. He et. al., `Deep Residual Learning for Image Recognition
    <https://arxiv.org/abs/1512.03385>`_
    Args:
        pretrained_model (str): the destination of the pre-trained
            chainer model serialized as a ``.npz`` file.
            If this argument is specified as ``auto``,
            it automatically loads and converts the caffemodel from
            ``$CHAINER_DATASET_ROOT/pfnet/chainer/models/ResNet-50-model.caffemodel``,
            where ``$CHAINER_DATASET_ROOT`` is set as
            ``$HOME/.chainer/dataset`` unless you specify another value
            by modifying the environment variable. Note that in this case the
            converted chainer model is stored on the same directory and
            automatically used from the next time.
            If this argument is specified as ``None``, all the parameters
            are not initialized by the pre-trained model, but the default
            initializer used in the original paper, i.e.,
            ``chainer.initializers.HeNormal(scale=1.0)``.
    Attributes:
        ~ResNet50Layers.available_layers (list of str): The list of available
            layer names used by ``__call__`` and ``extract`` methods.
    """

    #def __init__(self, n_out,entropy_weight,n_layers=50,pretrained_model='auto'):
    def __init__(self,pretrained_model='auto'):
        if pretrained_model == 'auto':
            pretrained_model = 'ResNet-50-model.caffemodel'
        #n_layers=50
        super(ResNet50Layers, self).__init__(pretrained_model, 50)
        #super(ResNet50Layers, self).__init__(n_out,entropy_weight,n_layers,pretrained_model)

class ResNet50Layers_transfer(chainer.Chain):

    def __init__(self, n_out,entropy_weight):
        super(ResNet50Layers_transfer, self).__init__()
        with self.init_scope():
            self.base = ResNet50Layers(pretrained_model='auto')
            self.fc7  = Linear(None,n_out)
        #self.base=base
        self.entropy_weight=entropy_weight
        self.n_out=n_out
        #self.fc7=fc7

    def __call__(self, x,t):
        functions_ordered_dict=self.base.functions()
        #print('x',x.shape,type(x))
        #print('t',t.shape,type(t))
        for key,funcs in functions_ordered_dict.items():
            for func in funcs:
                #print('func',func)
                x = func(x)#retrun fc6
                #print(func)
                #print('x',x.shape,type(x))
                #print('--------')
        #h = self.base.calc_activation(x, layers=['fc6'])
        #print('x',x.shape,type(x))
        y = self.fc7(x)
        #print('y',y.shape,type(y))
        #print(y,y.shape)
        #print(t,t.shape)
        #cw = cupy.array([1, self.entropy_weight]).astype(cupy.float32)
        cw = cupy.array([self.entropy_weight[0]]).astype(cupy.float32)
        if len(self.entropy_weight) > 1:
            for k in range(len(self.entropy_weight)-1):
                tmp_weight=self.entropy_weight[k+1]
                tmp=cupy.array([tmp_weight]).astype(cupy.float32)
                cw = cupy.concatenate([cw,tmp])
        loss, accuracy = F.softmax_cross_entropy(y, t,class_weight=cw), F.accuracy(y, t)
        f1_score=F.f1_score(y,t)
        #loss, accuracy = F.softmax_cross_entropy(h, t,class_weight=cupy.ndarray([1,10]).astype(cupy.float32)), F.accuracy(h, t,class_weight=cupy.ndarray([1,10]).astype(cupy.float32))
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        return y, loss, accuracy,f1_score

    def calc_activation(self, x, layers=None, **kwargs):
        """calc_activation(self, x, layers=['fc6'])
        Computes all the feature maps specified by ``layers``.
        .. warning::
           ``test`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', train)``.
           See :func:`chainer.using_config`.
        Args:
            x (~chainer.Variable): Input variable. It should be prepared by
                ``prepare`` function.
            layers (list of str): The list of layer names you want to extract.
        Returns:
            Dictionary of ~chainer.Variable: A directory in which
            the key contains the layer name and the value contains
            the corresponding feature map variable.
        """

        if layers is None:
            layers = ['fc6']

        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

        h = x
        activations = {}
        target_layers = set(layers)
        functions_ordered_dict=self.base.functions()
        for key, funcs in functions_ordered_dict.items():
            #print(key)
            #print(funcs)
            #print('h',h.shape,type(h))
            if len(target_layers) == 0:
                break
            for func in funcs:
                #print(func)
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        #print('calc_activation,finished')
        #print(activations)
        return activations

    def extract(self, images, layers=None,size=(224, 224), **kwargs):
        """extract(self, images, layers=['pool5'], size=(224, 224))
        Extracts all the feature maps of given images.
        The difference of directly executing ``__call__`` is that
        it directly accepts images as an input and automatically
        transforms them to a proper variable. That is,
        it is also interpreted as a shortcut method that implicitly calls
        ``prepare`` and ``__call__`` functions.
        .. warning::
           ``test`` and ``volatile`` arguments are not supported anymore since
           v2.
           Instead, use ``chainer.using_config('train', train)`` and
           ``chainer.using_config('enable_backprop', not volatile)``
           respectively.
           See :func:`chainer.using_config`.
        Args:
            images (iterable of PIL.Image or numpy.ndarray): Input images.
            layers (list of str): The list of layer names you want to extract.
            size (pair of ints): The resolution of resized images used as
                an input of CNN. All the given images are not resized
                if this argument is ``None``, but the resolutions of
                all the images should be the same.
        Returns:
            Dictionary of ~chainer.Variable: A directory in which
            the key contains the layer name and the value contains
            the corresponding feature map variable.
        """

        if layers is None:
            layers = ['res5']#use in i.e GradCAM

        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
            'Use chainer.using_config',
            volatile='volatile argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

        #x = concat_examples([prepare(img, size=size) for img in images])
        x = images
        #print(x)
        #print('extracting...',x.dtype,x.shape)
        #print(type(x))
        #x = Variable(self.base.xp.asarray(x))
        x = Variable(x)
        #print('extract_finished')
        #print(x.shape,type(x))
        return self.calc_activation(x,layers=layers)

    #def predict(self, images, oversample=True):
    def predict(self, images, oversample=False):
        """Computes all the probabilities of given images.
        Args:
            images (iterable of PIL.Image or numpy.ndarray): Input images.
                When you specify a color image as a :class:`numpy.ndarray`,
                make sure that color order is RGB.
            oversample (bool): If ``True``, it averages results across
                center, corners, and mirrors. Otherwise, it uses only the
                center.
        Returns:
            ~chainer.Variable: Output that contains the class probabilities
            of given images.
        """
        #x = concat_examples([prepare(img, size=(256, 256)) for img in images])
        #x = concat_examples([prepare(img, size=(224, 224)) for img in images])
        # Use no_backprop_mode to reduce memory consumption
        x = images
        #print(x)
        with function.no_backprop_mode(), chainer.using_config('train', False):
            #x = Variable(x)
            #x = Variable(self.base.xp.asarray(x))
            #print('predicting',x.dtype,x.shape,type(x))
            y = self.extract(x, layers=['fc6'])['fc6']
            #print('y',y.shape,type(y))
            y = self.fc7(y)
            y = softmax(y)#probability calc
        return y

class BuildingBlock(link.Chain):

    """A building block that consists of several Bottleneck layers.
    Args:
        n_layer (int): Number of layers used in the building block.
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, initialW=None):
        super(BuildingBlock, self).__init__()
        with self.init_scope():
            self.a = BottleneckA(
                in_channels, mid_channels, out_channels, stride, initialW)
            self._forward = ["a"]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = BottleneckB(out_channels, mid_channels, initialW)
                setattr(self, name, bottleneck)
                self._forward.append(name)

    def __call__(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
        return x

    @property
    def forward(self):
        return [getattr(self, name) for name in self._forward]


class BottleneckA(link.Chain):

    """A bottleneck layer that reduces the resolution of the feature map.
    Args:
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, initialW=None):
        super(BottleneckA, self).__init__()
        with self.init_scope():
            self.in_channels=in_channels
            self.mid_channels=mid_channels
            self.out_channels=out_channels
            self.conv1 = Convolution2D(
                in_channels, mid_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn1 = BatchNormalization(mid_channels)
            self.conv2 = Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = BatchNormalization(mid_channels)
            self.conv3 = Convolution2D(
                mid_channels, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = BatchNormalization(out_channels)
            self.conv4 = Convolution2D(
                in_channels, out_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn4 = BatchNormalization(out_channels)

    def __call__(self, x):
        h1 = relu(self.bn1(self.conv1(x)))
        self.cam_a1=h1
        h1 = relu(self.bn2(self.conv2(h1)))
        self.cam_a2=h1
        h1 = self.bn3(self.conv3(h1))
        self.cam_a3=h1
        h2 = self.bn4(self.conv4(x))
        self.cam_a4=h2
        return relu(h1 + h2)


class BottleneckB(link.Chain):

    """A bottleneck layer that maintains the resolution of the feature map.
    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, initialW=None):
        super(BottleneckB, self).__init__()
        with self.init_scope():
            self.conv1 = Convolution2D(
                in_channels, mid_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn1 = BatchNormalization(mid_channels)
            self.conv2 = Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = BatchNormalization(mid_channels)
            self.conv3 = Convolution2D(
                mid_channels, in_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = BatchNormalization(in_channels)

    def __call__(self, x):
        h = relu(self.bn1(self.conv1(x)))
        self.cam_b1=h
        h = relu(self.bn2(self.conv2(h)))
        self.cam_b2=h
        h = self.bn3(self.conv3(h))
        self.cam_b3=h
        return relu(h + x)


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = average_pooling_2d(x, (rows, cols), stride=1)
    h = reshape(h, (n, channel))
    return h


def _transfer_components(src, dst_conv, dst_bn, bname, cname):
    src_conv = getattr(src, 'res{}_branch{}'.format(bname, cname))
    src_bn = getattr(src, 'bn{}_branch{}'.format(bname, cname))
    src_scale = getattr(src, 'scale{}_branch{}'.format(bname, cname))
    dst_conv.W.data[:] = src_conv.W.data
    dst_bn.avg_mean[:] = src_bn.avg_mean
    dst_bn.avg_var[:] = src_bn.avg_var
    dst_bn.gamma.data[:] = src_scale.W.data
    dst_bn.beta.data[:] = src_scale.bias.b.data


def _transfer_bottleneckA(src, dst, name):
    _transfer_components(src, dst.conv1, dst.bn1, name, '2a')
    _transfer_components(src, dst.conv2, dst.bn2, name, '2b')
    _transfer_components(src, dst.conv3, dst.bn3, name, '2c')
    _transfer_components(src, dst.conv4, dst.bn4, name, '1')


def _transfer_bottleneckB(src, dst, name):
    _transfer_components(src, dst.conv1, dst.bn1, name, '2a')
    _transfer_components(src, dst.conv2, dst.bn2, name, '2b')
    _transfer_components(src, dst.conv3, dst.bn3, name, '2c')


def _transfer_block(src, dst, names):
    _transfer_bottleneckA(src, dst.a, names[0])
    for i, name in enumerate(names[1:]):
        dst_bottleneckB = getattr(dst, 'b{}'.format(i + 1))
        _transfer_bottleneckB(src, dst_bottleneckB, name)


def _transfer_resnet50(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.conv1.b.data[:] = src.conv1.b.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3, ['3a', '3b', '3c', '3d'])
    _transfer_block(src, dst.res4, ['4a', '4b', '4c', '4d', '4e', '4f'])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc6.W.data[:] = src.fc1000.W.data
    dst.fc6.b.data[:] = src.fc1000.b.data


def _transfer_resnet101(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3, ['3a', '3b1', '3b2', '3b3'])
    _transfer_block(src, dst.res4,
                    ['4a'] + ['4b{}'.format(i) for i in range(1, 23)])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc6.W.data[:] = src.fc1000.W.data
    dst.fc6.b.data[:] = src.fc1000.b.data


def _transfer_resnet152(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3,
                    ['3a'] + ['3b{}'.format(i) for i in range(1, 8)])
    _transfer_block(src, dst.res4,
                    ['4a'] + ['4b{}'.format(i) for i in range(1, 36)])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc6.W.data[:] = src.fc1000.W.data
    dst.fc6.b.data[:] = src.fc1000.b.data


def _make_npz(path_npz, path_caffemodel, model, n_layers):
    print('Now loading caffemodel (usually it may take few minutes)')
    if not os.path.exists(path_caffemodel):
        raise IOError(
            'The pre-trained caffemodel does not exist. Please download it '
            'from \'https://github.com/KaimingHe/deep-residual-networks\', '
            'and place it on {}'.format(path_caffemodel))

    ResNetLayers.convert_caffemodel_to_npz(path_caffemodel, path_npz, n_layers)
    npz.load_npz(path_npz, model)
    return model


def _retrieve(n_layers, name_npz, name_caffemodel, model):
    root = download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name_npz)
    path_caffemodel = os.path.join(root, name_caffemodel)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, path_caffemodel, model, n_layers),
        lambda path: npz.load_npz(path, model))