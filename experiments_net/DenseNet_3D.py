import tensorflow as tf

from tensorflow.keras.layers import Dense, Activation, Conv3D, MaxPool3D, Dropout, concatenate, BatchNormalization, GlobalAveragePooling3D
from tensorflow.keras.regularizers import l2


class CBL(tf.keras.Model):
    def __init__(self, filters, kernel=(3, 3, 3), drop_rate=0., weight_decay=0.005):
        super(CBL, self).__init__(name='cbl_block')
        self.drop_rate = drop_rate
        self.conv3d = Conv3D(filters, kernel,
                             kernel_initializer='he_normal',
                             padding='same',
                             use_bias=False,
                             kernel_regularizer=l2(weight_decay))

        self.bn = BatchNormalization()
        self.activation = Activation('relu')
        self.dropout = Dropout(drop_rate)

    def call(self, input_tensor, training=True):
        x = self.conv3d(input_tensor)
        x = self.bn(x, training=training)
        x = self.activation(x)
        if self.drop_rate:
            x = self.dropout(x)
        return x


class DenseBlock(tf.keras.Model):
    def __init__(self, growth_rate, internal_layers=4, drop_rate=0., weight_decay=0.005):
        super(DenseBlock, self).__init__(name='denseblock')
        self.internal_layers = internal_layers
        self.cbl = [CBL(growth_rate, drop_rate=drop_rate, weight_decay=weight_decay) for _ in range(self.internal_layers)]

    def call(self, input_tensor, training=True):
        list_feat = []
        list_feat.append(input_tensor)
        x = self.cbl[0](input_tensor, training=training)
        list_feat.append(x)
        x = concatenate(list_feat, axis=-1)
        for i in range(1, self.internal_layers):
            x = self.cbl[i](x, training=training)
            list_feat.append(x)
            x = concatenate(list_feat, axis=-1)

        return x


class DenseNet_3D_Model(tf.keras.Model):
    def __init__(self, num_class=2, weight_decay=0.005, drop_rate=0.):
        super(DenseNet_3D_Model, self).__init__()

        # stage 1
        self.cbl_1 = CBL(filters=64)
        self.maxpool3d_1 = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')

        # stage 2
        self.denseblock_2 = DenseBlock(growth_rate=32, internal_layers=4, drop_rate=drop_rate)
        self.maxpool3d_2 = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')
        self.cbl_2 = CBL(filters=128, kernel=(1, 1, 1), drop_rate=drop_rate)

        # stage 3
        self.denseblock_3 = DenseBlock(growth_rate=32, internal_layers=4, drop_rate=drop_rate)
        self.maxpool3d_3 = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')
        self.cbl_3 = CBL(filters=128, kernel=(1, 1, 1), drop_rate=drop_rate)

        # stage 4
        self.denseblock_4 = DenseBlock(growth_rate=64, internal_layers=4, drop_rate=drop_rate)
        self.maxpool3d_4 = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')
        self.cbl_4 = CBL(filters=256, kernel=(1, 1, 1), drop_rate=drop_rate)

        # stage 5
        self.denseblock_5 = DenseBlock(growth_rate=64, internal_layers=4, drop_rate=drop_rate)
        self.cbl_5 = CBL(filters=256, kernel=(1, 1, 1), drop_rate=drop_rate)

        self.globalavgpool3d = GlobalAveragePooling3D()
        self.dense = Dense(num_class,
                           activation='softmax',
                           kernel_regularizer=l2(weight_decay),
                           bias_regularizer=l2(weight_decay))

    def call(self, input_tensor, training=True):
        x = self.cbl_1(input_tensor, training=training)
        x = self.maxpool3d_1(x)

        x = self.denseblock_2(x, training=training)
        x = self.maxpool3d_2(x)
        x = self.cbl_2(x, training=training)

        x = self.denseblock_3(x, training=training)
        x = self.maxpool3d_3(x)
        x = self.cbl_3(x, training=training)

        x = self.denseblock_4(x, training=training)
        x = self.maxpool3d_4(x)
        x = self.cbl_4(x, training=training)

        x = self.denseblock_5(x, training=training)
        x = self.cbl_5(x, training=training)

        x = self.globalavgpool3d(x)
        x = self.dense(x)

        return x
