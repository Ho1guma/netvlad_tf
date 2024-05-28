
import tensorflow as tf


# def netVLAD(inputs, num_clusters, assign_weight_initializer=None,
#             cluster_initializer=None, skip_postnorm=False):
#     ''' skip_postnorm: Only there for compatibility with mat files. '''
#     K = num_clusters
#     D = inputs.get_shape()[-1]
#     s = tf.keras.layers.Conv2D(K, 1, use_bias=False,
#                          kernel_initializer=assign_weight_initializer)(inputs)
#     a = tf.keras.activations.softmax(s, axis=-1)
#     a = tf.expand_dims(a, -2) #(b,8,10,1,64)
#     shape = (1, 1, 1, 512, 32)
#     initializer = tf.initializers.he_normal()
#     C = tf.Variable(lambda : initializer(shape=(1, 1, 1, 512, 32), dtype=inputs.dtype))
#     v = tf.expand_dims(inputs, -1) + C
#     v = a * v
#     v = tf.reduce_sum(v, axis=[1, 2])
#     v = tf.transpose(v, perm=[0, 2, 1])
#     v = v / tf.sqrt(tf.math.reduce_sum(v ** 2, axis=-1, keepdims=True)
#                         + 1e-12)
#     v = tf.transpose(v, perm=[0, 2, 1])
#     v = tf.keras.layers.Flatten()(v)
#     v = v / tf.sqrt(tf.math.reduce_sum(v ** 2, axis=-1, keepdims=True)
#                     + 1e-12)

#     return v

class NetVLAD(tf.keras.Model):
    def __init__(self, num_clusters=32):
        super(NetVLAD, self).__init__()

        self.conv2 = tf.keras.layers.Conv2D(num_clusters, 1, use_bias=False)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

        self.initializer = tf.initializers.he_normal()
        self.C = tf.Variable(lambda: self.initializer(shape=(1, 1, 1, 512, 32)),trainable=True)
        self.add = tf.keras.layers.Add()
        self.mul = tf.keras.layers.Multiply()
        self.flatten = tf.keras.layers.Flatten()


    def call(self, x, training=None, mask=None):
        s = self.conv2(x)
        a = self.softmax(s)
        a = tf.reshape(a, shape=[-1, a.shape[1], a.shape[2], 1, a.shape[3]])
        x = tf.reshape(x, shape=[-1, x.shape[1], x.shape[2], x.shape[3], 1])
        v = self.add([x,self.C])
        v = self.mul([a,v])
        v = tf.reduce_sum(v, axis=[1, 2])
        #v = tf.transpose(v, perm=[0, 2, 1])
        output = self.flatten(v)

        return output



def netvlad(inputs=None, num_clusters=32):
    return NetVLAD(num_clusters)
