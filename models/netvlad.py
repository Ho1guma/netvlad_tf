
import tensorflow as tf
import config

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
        if config.kernel:
            self.padding = tf.keras.layers.ZeroPadding2D(padding=(1,1))
            self.conv3 = tf.keras.layers.Conv2D(512, config.kernel, strides=2, padding="valid", use_bias=False)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

        self.initializer = tf.initializers.he_normal()
        self.C = tf.Variable(lambda: self.initializer(shape=(1, 1, 1, 512, 32)),trainable=True)
        self.add = tf.keras.layers.Add()
        self.mul = tf.keras.layers.Multiply()
        self.flatten = tf.keras.layers.Flatten()

    def normalize_tensor(self,input, p, dim, eps = 1e-12):
        norm = tf.norm(input, ord=p, axis=dim, keepdims=True)
        norm = tf.clip_by_value(norm, clip_value_min=eps, clip_value_max=tf.reduce_max(norm))
        norm = tf.broadcast_to(norm, tf.shape(input))
        normalized_input = input / norm
        return normalized_input


    def call(self, x, training=None, mask=None):
        a=1
        # x = tf.identity(x)
        # x = self.normalize_tensor(x,p=2,dim=3) # norm 1: 삭제 불가!
        if config.kernel:
            x = self.padding(x)
            x = self.conv3(x)
        s = self.conv2(x)
        a = self.softmax(s)
        a = tf.keras.backend.expand_dims(a, -2)  # (b,8,10,1,64)
        x = tf.keras.backend.expand_dims(x, -1)
        v = self.add([x,self.C])
        v = self.mul([a,v])
        v = tf.reduce_sum(v, axis=[1, 2])
        #v = tf.transpose(v, perm=[0, 2, 1])
        v= self.normalize_tensor(v, p=2, dim=2) # norm 2
        v = self.flatten(v)
        output = self.normalize_tensor(v, p=2, dim=1)
        return output



def netvlad(inputs=None, num_clusters=32):
    return NetVLAD(num_clusters)
