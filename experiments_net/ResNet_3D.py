import tensorflow as tf

from tensorflow.keras.layers import Dense, add, Activation, Conv3D, MaxPooling3D, Dropout, BatchNormalization, GlobalAveragePooling3D
from tensorflow.keras.regularizers import l2


class BLC(tf.keras.Model):
    def __init__(self, filters, kernel=(3, 3, 3), weight_decay=0.005):
        super(BLC, self).__init__(name='blc_block')

        self.conv3d = Conv3D(filters, kernel,
                             kernel_initializer='he_normal',
                             padding='same',
                             use_bias=False,
                             kernel_regularizer=l2(weight_decay))

        self.bn = BatchNormalization(axis=-1, epsilon=1.1e-5)
        self.activation = Activation('relu')

    def call(self, input_tensor, training=True):
        x = self.bn(input_tensor, training=training)
        x = self.activation(x)
        x = self.conv3d(x)

        return x


class ResidualBlock(tf.keras.Model):
    def __init__(self, filters, drop_rate=0., weight_decay=0.005):
        super(ResidualBlock, self).__init__()

        self.drop_rate = drop_rate

        self.blc_1 = BLC(filters*4, kernel=(1, 1, 1))
        self.dropout_1 = Dropout(drop_rate)

        self.blc_2 = BLC(filters, kernel=(3, 3, 3))
        self.dropout_2 = Dropout(drop_rate)

        self.conv3d_3 = Conv3D(filters*4, (1, 1, 1),
                               kernel_initializer='he_normal',
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))

    def call(self, input_tensor, training=True):
        x = self.blc_1(input_tensor, training=training)
        if self.drop_rate:
            x = self.dropout_1(x)
        x = self.blc_2(x, training=training)
        if self.drop_rate:
            x = self.dropout_2(x)
        x = self.conv3d_3(x)

        return x


class ResNet_3D_Model(tf.keras.Model):
    def __init__(self, num_class=2, drop_rate=0., weight_decay=0.005):
        super(ResNet_3D_Model, self).__init__()
        # 64, 128, 128, 128, 256, 256, 256, 256, 256
        # stage 1
        self.conv3d_1 = Conv3D(64, (3, 3, 3),
                               kernel_initializer='he_normal',
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))
        self.bn_1 = BatchNormalization(axis=-1, epsilon=1.1e-5)
        self.activition_1 = Activation('relu')
        self.maxpooling3d_1 = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')

        # stage 2
        self.conv3d_2 = Conv3D(128, (1, 1, 1),
                               kernel_initializer='he_normal',
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))
        self.residualblock_2 = ResidualBlock(32, drop_rate=drop_rate, weight_decay=weight_decay)
        self.maxpooling3d_2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')

        # stage 3
        self.residualblock_3 = ResidualBlock(32, drop_rate=drop_rate, weight_decay=weight_decay)
        self.maxpooling3d_3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')

        # stage 4
        self.bn_4 = BatchNormalization(axis=-1, epsilon=1.1e-5)
        self.activation_4 = Activation('relu')
        self.conv3d_4 = Conv3D(256, (1, 1, 1),
                               kernel_initializer='he_normal',
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))
        self.residualblock_4_1 = ResidualBlock(64, drop_rate=drop_rate, weight_decay=weight_decay)
        self.residualblock_4_2 = ResidualBlock(64, drop_rate=drop_rate, weight_decay=weight_decay)
        self.maxpooling3d_4 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')

        # stage 5
        self.residualblock_5_1 = ResidualBlock(64, drop_rate=drop_rate, weight_decay=weight_decay)
        self.residualblock_5_2 = ResidualBlock(64, drop_rate=drop_rate, weight_decay=weight_decay)
        self.maxpooling3d_5 = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')

        # output 6
        self.bn_6 = BatchNormalization(axis=-1, epsilon=1.1e-5)
        self.activation_6 =Activation('relu')
        self.globalavgpooling3d_6 = GlobalAveragePooling3D()
        self.dense_6 = Dense(num_class,
                             activation='softmax',
                             kernel_regularizer=l2(weight_decay),
                             bias_regularizer=l2(weight_decay))

    def call(self, input_tensor, training=True):
        # stage 1
        x = self.conv3d_1(input_tensor)
        x = self.bn_1(x, training=training)
        x = self.activition_1(x)
        x = self.maxpooling3d_1(x)

        # stage 2
        origin_x = self.conv3d_2(x)
        x = self.residualblock_2(x, training=training)
        origin_x = add([x, origin_x])
        origin_x = self.maxpooling3d_2(origin_x)

        # stage 3
        x = self.residualblock_3(origin_x, training=training)
        origin_x = add([x, origin_x])
        x = self.maxpooling3d_3(origin_x)

        # stage 4
        origin_x = self.bn_4(x, training=training)
        origin_x = self.activation_4(origin_x)
        origin_x = self.conv3d_4(origin_x)
        x = self.residualblock_4_1(x, training=training)
        origin_x = add([x, origin_x])
        x = self.residualblock_4_2(origin_x, training=training)
        origin_x = add([x, origin_x])
        origin_x = self.maxpooling3d_4(origin_x)

        # stage 5
        x = self.residualblock_5_1(origin_x, training=training)
        origin_x = add([x, origin_x])
        x = self.residualblock_5_2(origin_x, training=training)
        origin_x = add([x, origin_x])
        origin_x = self.maxpooling3d_5(origin_x)

        # output 6
        x = self.bn_6(origin_x, training=training)
        x = self.activation_6(x)
        x = self.globalavgpooling3d_6(x)
        # tf.print(tf.shape(x))

        x = self.dense_6(x)

        return x







