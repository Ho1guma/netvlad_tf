import tensorflow as tf
from config import NUM_CLASSES
from models.residual_block import make_basic_block_layer, make_bottleneck_layer

def netVLAD(inputs, num_clusters, assign_weight_initializer=None,
            cluster_initializer=None, skip_postnorm=False):
    ''' skip_postnorm: Only there for compatibility with mat files. '''
    K = num_clusters
    D = inputs.get_shape()[-1]
    s = tf.keras.layers.Conv2D(K, 1, use_bias=False,
                         kernel_initializer=assign_weight_initializer)(inputs)
    a = tf.keras.activations.softmax(s, axis=-1)
    a = tf.expand_dims(a, -2) #(b,8,10,1,64)
    shape = (1, 1, 1, 512, 32)
    initializer = tf.initializers.he_normal()
    C = tf.Variable(lambda : initializer(shape=(1, 1, 1, 512, 32), dtype=inputs.dtype))
    v = tf.expand_dims(inputs, -1) + C
    v = a * v
    v = tf.reduce_sum(v, axis=[1, 2])
    v = tf.transpose(v, perm=[0, 2, 1])
    v = v / tf.sqrt(tf.math.reduce_sum(v ** 2, axis=-1, keepdims=True)
                        + 1e-12)
    v = tf.transpose(v, perm=[0, 2, 1])
    v = tf.keras.layers.Flatten()(v)
    v = v / tf.sqrt(tf.math.reduce_sum(v ** 2, axis=-1, keepdims=True)
                    + 1e-12)


    return v

def matconvnetNormalize(inputs, epsilon):
    return inputs / tf.sqrt(tf.reduce_sum(inputs ** 2, axis=-1, keep_dims=True)
                            + epsilon)

class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)
        self.conv2 = tf.keras.layers.Conv2D(32, 1, use_bias=False)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

        self.initializer = tf.initializers.he_normal()
        self.C = tf.Variable(lambda: self.initializer(shape=(1, 1, 1, 512, 32)))
        self.add = tf.keras.layers.Add()
        self.mul = tf.keras.layers.Multiply()
        self.flatten = tf.keras.layers.Flatten()
        #self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        #self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training) #(b,8,10,512)
        #tf.keras.backend.l2_normalize(x, axis=None)
        s = self.conv2(x)
        a = self.softmax(s)
        a = tf.keras.backend.expand_dims(a, -2)  # (b,8,10,1,64)
        x = tf.keras.backend.expand_dims(x, -1)
        v = self.add([x,self.C])
        v = self.mul([a,v])
        v = tf.reduce_sum(v, axis=[1, 2])
        #v = tf.transpose(v, perm=[0, 2, 1])
        output = self.flatten(v)

        return output


class ResNetTypeII(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output


def resnet_18():
    return ResNetTypeI(layer_params=[2, 2, 2, 2])


def resnet_34():
    return ResNetTypeI(layer_params=[3, 4, 6, 3])


def resnet_50():
    return ResNetTypeII(layer_params=[3, 4, 6, 3])


def resnet_101():
    return ResNetTypeII(layer_params=[3, 4, 23, 3])


def resnet_152():
    return ResNetTypeII(layer_params=[3, 8, 36, 3])
