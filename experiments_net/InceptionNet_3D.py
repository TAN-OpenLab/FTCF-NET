import tensorflow as tf

from tensorflow.keras.layers import Dense, Activation, Conv3D, MaxPooling3D, Dropout, concatenate, BatchNormalization, AveragePooling3D, GlobalAveragePooling3D
from tensorflow.keras.regularizers import l2


class CBL(tf.keras.Model):
    def __init__(self, filters, kernel=(3, 3, 3), drop_rate=0., weight_decay=0.005):
        super(CBL, self).__init__()

        self.drop_rate = drop_rate
        self.conv3d = Conv3D(filters, kernel,
                             kernel_initializer='he_normal',
                             padding="same",
                             use_bias=False,
                             kernel_regularizer=l2(weight_decay))
        self.bn = BatchNormalization(axis=-1, epsilon=1.1e-5)
        self.activation = Activation('relu')
        self.dropout = Dropout(drop_rate)

    def call(self, input_tensor):
        x = self.conv3d(input_tensor)
        x = self.bn(x)
        x = self.activation(x)
        if self.drop_rate:
            x = self.dropout(x)

        return x


class InceptionBlock(tf.keras.Model):
    def __init__(self, filters, kernels, pools, drop_rate=0., weight_decay=0.005):
        super(InceptionBlock, self).__init__()

        self.branch_1 = CBL(filters, kernel=(1, 1, 1), drop_rate=drop_rate, weight_decay=weight_decay)

        self.branch_2_1 = CBL(filters, kernel=(1, 1, 1), drop_rate=drop_rate, weight_decay=weight_decay)
        self.branch_2_2 = CBL(filters, kernel=kernels[0], drop_rate=drop_rate, weight_decay=weight_decay)

        self.branch_3_1 = CBL(filters, kernel=(1, 1, 1), drop_rate=drop_rate, weight_decay=weight_decay)
        self.branch_3_2 = CBL(filters, kernel=kernels[1], drop_rate=drop_rate, weight_decay=weight_decay)
        self.branch_3_3 = CBL(filters, kernel=kernels[2], drop_rate=drop_rate, weight_decay=weight_decay)

        self.branch_4_1 = AveragePooling3D(pool_size=pools[0], strides=pools[1], padding='same')
        self.branch_4_2 = CBL(filters, kernel=(1, 1, 1), drop_rate=drop_rate, weight_decay=weight_decay)

    def call(self, input_tensor):
        branch_1 = self.branch_1(input_tensor)

        branch_2 = self.branch_2_1(input_tensor)
        branch_2 = self.branch_2_2(branch_2)

        branch_3 = self.branch_3_1(input_tensor)
        branch_3 = self.branch_3_2(branch_3)
        branch_3 = self.branch_3_3(branch_3)

        branch_4 = self.branch_4_1(input_tensor)
        branch_4 = self.branch_4_2(branch_4)

        x = concatenate([branch_1, branch_2, branch_3, branch_4], axis=-1)
        return x


class InceptionNet_3D_Model(tf.keras.Model):
    def __init__(self, num_class=2, drop_rate=0.2, weight_decay=0.005):
        super(InceptionNet_3D_Model, self).__init__()

        self.CBL_0 = CBL(64)
        self.maxpooling3d_0 = MaxPooling3D((2, 2, 1), strides=(2, 2, 1), padding='same')

        # stage 1
        self.inception_1 = InceptionBlock(32, kernels=[(5, 5, 3), (3, 3, 3), (3, 3, 3)],
                                          pools=[(2, 2, 2), (1, 1, 1)],
                                          drop_rate=drop_rate,
                                          weight_decay=weight_decay)
        self.maxpooling3d_1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')

        # stage 2
        self.inception_2 = InceptionBlock(32, kernels=[(5, 5, 3), (3, 3, 3), (3, 3, 3)],
                                          pools=[(2, 2, 2), (1, 1, 1)],
                                          drop_rate=drop_rate,
                                          weight_decay=weight_decay)
        self.maxpooling3d_2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')

        # stage 3
        self.inception_3_1 = InceptionBlock(64, kernels=[(5, 5, 3), (7, 1, 3), (1, 7, 3)],
                                            pools=[(2, 2, 2), (1, 1, 1)],
                                            drop_rate=drop_rate,
                                            weight_decay=weight_decay)

        self.inception_3_2 = InceptionBlock(64, kernels=[(5, 5, 3), (7, 1, 3), (1, 7, 3)],
                                            pools=[(2, 2, 2), (1, 1, 1)],
                                            drop_rate=drop_rate,
                                            weight_decay=weight_decay)
        self.maxpooling3d_3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')

        # stage 4
        self.inception_4_1 = InceptionBlock(64, kernels=[(3, 3, 1), (7, 1, 1), (1, 7, 1)],
                                            pools=[(2, 2, 1), (1, 1, 1)],
                                            drop_rate=drop_rate,
                                            weight_decay=weight_decay)

        self.inception_4_2 = InceptionBlock(64, kernels=[(3, 3, 1), (7, 1, 1), (1, 7, 1)],
                                            pools=[(2, 2, 1), (1, 1, 1)],
                                            drop_rate=drop_rate,
                                            weight_decay=weight_decay)
        self.maxpooling3d_4 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')

        self.cbl_5 = CBL(256, (1, 1, 1))
        self.globalavgpooling3d_5 = GlobalAveragePooling3D()
        self.dense_5 = Dense(num_class,
                             activation='softmax',
                             kernel_regularizer=l2(weight_decay),
                             bias_regularizer=l2(weight_decay))

    def call(self, input_tensor):
        x = self.CBL_0(input_tensor)
        x = self.maxpooling3d_0(x)

        x = self.inception_1(x)
        x = self.maxpooling3d_1(x)

        x = self.inception_2(x)
        x = self.maxpooling3d_2(x)

        x = self.inception_3_1(x)
        x = self.inception_3_2(x)
        x = self.maxpooling3d_3(x)

        x = self.inception_4_1(x)
        x = self.inception_4_2(x)
        x = self.maxpooling3d_4(x)

        x = self.cbl_5(x)
        x = self.globalavgpooling3d_5(x)
        x = self.dense_5(x)

        return x
