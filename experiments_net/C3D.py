import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from tensorflow.keras.regularizers import l2


class C3D_Model(tf.keras.layers.Layer):
    def __init__(self, num_class=2, weight_decay=0.005):
        super(C3D_Model, self).__init__()
        # 64, 128, 128, 256, 256
        self.weight_decay = weight_decay
        self.conv3d_1 = Conv3D(64, (3, 3, 3),
                               strides=(1, 1, 1),
                               padding='same',
                               activation='relu',
                               kernel_regularizer=l2(self.weight_decay))
        self.maxpooling3d_1 = MaxPooling3D((2, 2, 1), strides=(2, 2, 1), padding='same')

        self.conv3d_2 = Conv3D(128, (3, 3, 3),
                               strides=(1, 1, 1),
                               padding='same',
                               activation='relu',
                               kernel_regularizer=l2(self.weight_decay))
        self.maxpooling3d_2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')

        self.conv3d_3 = Conv3D(128, (3, 3, 3),
                               strides=(1, 1, 1),
                               padding='same',
                               activation='relu',
                               kernel_regularizer=l2(self.weight_decay))
        self.maxpooling3d_3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')

        self.conv3d_4 = Conv3D(256, (3, 3, 3),
                               strides=(1, 1, 1),
                               padding='same',
                               activation='relu',
                               kernel_regularizer=l2(self.weight_decay))
        self.maxpooling3d_4 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')

        self.conv3d_5 = Conv3D(256, (3, 3, 3),
                               strides=(1, 1, 1),
                               padding='same',
                               activation='relu',
                               kernel_regularizer=l2(self.weight_decay))
        self.maxpooling3d_5 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')

        self.flatten = Flatten()
        self.dropout_0 = Dropout(0.5)
        self.dense_1 = Dense(1024, activation='relu', kernel_regularizer=l2(self.weight_decay))
        self.dropout_1 = Dropout(0.5)
        self.dense_2 = Dense(512, activation='relu', kernel_regularizer=l2(self.weight_decay))
        self.dropout_2 = Dropout(0.5)
        self.output_tensor = Dense(num_class, activation='softmax', kernel_regularizer=l2(self.weight_decay))

    def call(self, input_tensor, training=True):
        x = self.conv3d_1(input_tensor)
        x = self.maxpooling3d_1(x)
        x = self.conv3d_2(x)
        x = self.maxpooling3d_2(x)
        x = self.conv3d_3(x)
        x = self.maxpooling3d_3(x)
        x = self.conv3d_4(x)
        x = self.maxpooling3d_4(x)
        x = self.conv3d_5(x)
        x = self.maxpooling3d_5(x)
        x = self.flatten(x)
        x = self.dropout_0(x)
        x = self.dense_1(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)

        return self.output_tensor(x)
