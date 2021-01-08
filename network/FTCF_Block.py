import tensorflow as tf

from tensorflow.keras.layers import Conv3D, add, BatchNormalization, Conv2D
from tensorflow.keras.regularizers import l2


class FTFC_ExpandSubNet(tf.keras.Model):
    def __init__(self, filters=15, kernel_size=(3, 3), strides=(1, 1), weight_decay=0.005, dropout_rate=0.5):
        super(FTFC_ExpandSubNet, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        self.conv2d_1 = Conv2D(filters=filters*2,
                               kernel_size=self.kernel_size,
                               strides=self.strides,
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(self.weight_decay))
        self.bn_1 = BatchNormalization()

        self.conv2d_3 = Conv2D(filters=filters,
                               kernel_size=self.kernel_size,
                               strides=self.strides,
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(self.weight_decay))
        self.bn_3 = BatchNormalization()

    def mish(self, tensor):
        return tensor*tf.nn.tanh(tf.nn.softplus(tensor))

    def call(self, input_tensor, training=True):
        x = self.conv2d_1(input_tensor)
        x = self.bn_1(x, training=training)
        x = self.mish(x)

        x = self.conv2d_3(x)
        x = self.bn_3(x, training=training)
        x = self.mish(x)

        return x


class FTCF_Block(tf.keras.layers.Layer):
    def __init__(self, filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), weight_decay=0.005):
        super(FTCF_Block, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.weight_decay = weight_decay

    def build(self, input_shape):
        self.T = input_shape[-2]
        self.channels = input_shape[-1]
        self.conv2d_list = []
        self.shape = input_shape
        self.small_full_temproal_network = [FTFC_ExpandSubNet(filters=self.T) for _ in range(self.channels)]

        self.conv3d_up = Conv3D(
            filters=self.filters,
            kernel_size=(self.kernel_size[0], self.kernel_size[1], 1),
            strides=self.strides,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(self.weight_decay),
        )
        self.conv3d_cp = Conv3D(
            filters=self.filters,
            kernel_size=(1, 1, 1),
            strides=self.strides,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(self.weight_decay),
        )
        self.bn_cp = BatchNormalization()
        self.bn_up = BatchNormalization()

    def mish(self, tensor):
        return tensor*tf.nn.tanh(tf.nn.softplus(tensor))

    def call(self, input_tensor, training=True):
        x_up = self.conv3d_up(input_tensor)
        x_up = self.bn_up(x_up, training=training)

        x = tf.transpose(input_tensor, (4, 0, 1, 2, 3))
        for i in range(self.channels):
            t = self.small_full_temproal_network[i](x[i], training=training)
            x = tf.tensor_scatter_nd_update(x, [[i]], tf.expand_dims(t, axis=0))
        x = tf.transpose(x, (1, 2, 3, 4, 0))
        x = self.conv3d_cp(x)
        x = self.bn_cp(x, training=training)

        x = add([x, x_up])
        x = self.mish(x)

        return x
