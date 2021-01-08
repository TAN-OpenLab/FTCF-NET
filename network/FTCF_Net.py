import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv3D, AveragePooling3D, Dropout, concatenate, GlobalAveragePooling3D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import Xception

from network.FTCF_Block import FTCF_Block
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class CBM(tf.keras.Model):
    def __init__(self, filters, kernel=(1, 1, 1), stride=(1, 1, 1), padding='same'):
        super(CBM, self).__init__()

        self.conv3d = Conv3D(filters, kernel,
                             strides=stride,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(0.005),
                             padding=padding)
        self.bn = BatchNormalization()

    def mish(self, x):
        return x * tf.nn.tanh(tf.nn.softplus(x))

    def call(self, input_tensor, training=True):
        x = self.conv3d(input_tensor)
        x = self.bn(x, training=training)
        x = self.mish(x)

        return x


class FTCF_Net(tf.keras.Model):
    def __init__(self, layer_name='add_3', weight_decay=0.005):
        super(FTCF_Net, self).__init__()
        self.weight_decay = weight_decay
        self.layer_name = layer_name
        self.basemodel = Xception(
            weights='imagenet',
            input_shape=(224, 224, 3),
            include_top=False
        )

        # Feature Extractor for Every Frames
        self.backbone = tf.keras.Model(inputs=self.basemodel.input,
                                       outputs=self.basemodel.get_layer(self.layer_name).output)
        self.backbone.trainable = False

        self.cbm1 = CBM(filters=364, kernel=(1, 1, 2))

        # Fusion Stage 1
        self.ftcf1 = FTCF_Block(32)
        self.avgpooling3d_1 = AveragePooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')

        # Fusion Stage 2
        self.ftcf2 = FTCF_Block(32)
        self.avgpooling3d_2 = AveragePooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')

        # Fusion Stage 3
        self.ftcf3 = FTCF_Block(64)
        self.avgpooling3d_3 = AveragePooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')

        self.cbm2 = CBM(filters=128, kernel=(2, 2, 2), padding='valid')
        self.globalpooling3d = GlobalAveragePooling3D()

        self.dense_2 = Dense(512, activation='relu', kernel_regularizer=l2(self.weight_decay))
        self.dropout_2 = Dropout(0.5)
        self.output_tensor = Dense(2, activation='softmax', kernel_regularizer=l2(self.weight_decay))

    def mish(self, x):
        return x * tf.nn.tanh(tf.nn.softplus(x))

    def call(self, input_tensor, training=True):
        single_frames = []
        for i in range(15):
            frame = self.backbone(input_tensor[:, :, :, i, :], training=False)
            frame = tf.expand_dims(frame, axis=3)
            single_frames.append(frame)
        x = concatenate(single_frames, axis=3)
        x = self.mish(x)

        x = self.cbm1(x, training=training)

        x = self.ftcf1(x, training=training)
        x = self.avgpooling3d_1(x)

        x = self.ftcf2(x, training=training)
        x = self.avgpooling3d_2(x)

        x = self.ftcf3(x, training=training)
        x = self.avgpooling3d_3(x)

        x = self.cbm2(x, training=training)
        x = self.globalpooling3d(x)

        x = self.dense_2(x)
        x = self.dropout_2(x)

        return self.output_tensor(x)
