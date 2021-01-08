import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv3D, MaxPool3D, Dropout, concatenate, BatchNormalization, AvgPool3D


class CBL(tf.keras.Model):
    def __init__(self, filters, kernel=(1, 1, 1), stride=(1, 1, 1), activation_fn=tf.nn.relu, use_bn=True, use_bias=False):
        super(CBL, self).__init__()

        self.use_bn = use_bn
        self.conv3d = Conv3D(filters, kernel,
                             strides=stride,
                             kernel_initializer='he_normal',
                             padding="same",
                             use_bias=use_bias)
        self.bn = BatchNormalization(axis=-1, epsilon=1.1e-5)
        self.activation = activation_fn

    def call(self, input_tensor, training=True):
        x = self.conv3d(input_tensor)
        if self.use_bn:
            x = self.bn(x, training=training)
        x = self.activation(x)

        return x


class MixedBlock(tf.keras.Model):
    def __init__(self, filters_list):
        super(MixedBlock, self).__init__()

        self.cbl_0 = CBL(filters_list[0], kernel=[1, 1, 1])

        self.cbl_1_1 = CBL(filters_list[1], kernel=[1, 1, 1])
        self.cbl_1_2 = CBL(filters_list[2], kernel=[3, 3, 3])

        self.cbl_2_1 = CBL(filters_list[3], kernel=[1, 1, 1])
        self.cbl_2_2 = CBL(filters_list[4], kernel=[3, 3, 3])

        self.maxpool3d_3 = MaxPool3D(pool_size=[3, 3, 3], strides=[1, 1, 1], padding='same')
        self.cbl_3 = CBL(filters_list[5], kernel=[1, 1, 1])

    def call(self, input_tensor, training=True):
        branch_0 = self.cbl_0(input_tensor, training=training)

        branch_1 = self.cbl_1_1(input_tensor, training=training)
        branch_1 = self.cbl_1_2(branch_1, training=training)

        branch_2 = self.cbl_2_1(input_tensor, training=training)
        branch_2 = self.cbl_2_2(branch_2, training=training)

        branch_3 = self.maxpool3d_3(input_tensor)
        branch_3 = self.cbl_3(branch_3, training=training)

        output_tensor = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
        return output_tensor


class Inflated_3D_Model(tf.keras.Model):
    def __init__(self, num_class=2, drop_rate=0.2):
        super(Inflated_3D_Model, self).__init__()

        # stage 1 head
        self.cbl_1_1 = CBL(filters=64, kernel=[7, 7, 7], stride=[2, 2, 2])
        self.maxpool3d_1_1 = MaxPool3D(pool_size=[3, 3, 1], strides=[2, 2, 1], padding='same')
        self.cbl_1_2 = CBL(filters=192, kernel=[3, 3, 3])
        self.maxpool3d_1_2 = MaxPool3D(pool_size=[3, 3, 1], strides=[2, 2, 1], padding='same')

        # stage 2
        self.mixedblock_2_1 = MixedBlock(filters_list=[64, 96, 128, 16, 32, 32])
        self.mixedblock_2_2 = MixedBlock(filters_list=[128, 128, 192, 32, 96, 64])
        self.maxpool3d_2 = MaxPool3D(pool_size=[3, 3, 3], strides=[2, 2, 2], padding='same')

        # stage 3
        self.mixedblock_3_1 = MixedBlock(filters_list=[192, 96, 208, 16, 48, 64])
        self.mixedblock_3_2 = MixedBlock(filters_list=[160, 112, 224, 24, 64, 64])
        self.mixedblock_3_3 = MixedBlock(filters_list=[128, 128, 256, 24, 64, 64])
        self.mixedblock_3_4 = MixedBlock(filters_list=[112, 144, 288, 32, 64, 64])
        self.mixedblock_3_5 = MixedBlock(filters_list=[256, 160, 320, 32, 128, 128])
        self.maxpool3d_3 = MaxPool3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same')

        # stage 4
        self.mixedblock_4_1 = MixedBlock(filters_list=[256, 160, 320, 32, 128, 128])
        self.mixedblock_4_2 = MixedBlock(filters_list=[384, 192, 384, 48, 128, 128])

        # stage 5
        self.avgpool3d_5 = AvgPool3D(pool_size=[7, 7, 2], strides=[1, 1, 1], padding='valid')

        self.dense_5_1 = Dense(512, activation='relu')
        self.dropout_5 = Dropout(drop_rate)
        self.dense_5_2 = Dense(num_class, activation='softmax')

    def call(self, input_tensor, training=True):
        x = self.cbl_1_1(input_tensor, training=training)
        x = self.maxpool3d_1_1(x)
        x = self.cbl_1_2(x, training=training)
        x = self.maxpool3d_1_2(x)

        x = self.mixedblock_2_1(x, training=training)
        x = self.mixedblock_2_2(x, training=training)
        x = self.maxpool3d_2(x)

        x = self.mixedblock_3_1(x, training=training)
        x = self.mixedblock_3_2(x, training=training)
        x = self.mixedblock_3_3(x, training=training)
        x = self.mixedblock_3_4(x, training=training)
        x = self.mixedblock_3_5(x, training=training)
        x = self.maxpool3d_3(x)

        x = self.mixedblock_4_1(x, training=training)
        x = self.mixedblock_4_2(x, training=training)
        x = self.avgpool3d_5(x)

        x = self.dropout_5(x)
        x = self.dense_5_1(x)
        x = self.dense_5_2(x)

        return x
