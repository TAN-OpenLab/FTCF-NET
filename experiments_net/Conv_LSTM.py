import tensorflow as tf

from tensorflow.keras.layers import Dense, BatchNormalization,RNN, LSTMCell, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.regularizers import l2


class Conv_LSTM_Model(tf.keras.Model):
    def __init__(self, units=128, num_class=2):
        super(Conv_LSTM_Model, self).__init__()

        self.lstm_1 = LSTMCell(units)
        self.cells = [self.lstm_1]
        self.rnn = RNN(self.cells, return_sequences=True)
        self.basemodel = VGG16(
            weights='imagenet',
            input_shape=(224, 224, 3),
            include_top=True
        )
        self.backbone = tf.keras.Model(inputs=self.basemodel.input,
                                       outputs=self.basemodel.get_layer('fc1').output)
        self.backbone.trainable = False
        self.bn = BatchNormalization()
        self.dense_1 = Dense(2048, activation='relu', kernel_regularizer=l2(0.15))
        self.dropout_1 = Dropout(0.2)
        self.dense_2 = Dense(1024, activation='relu', kernel_regularizer=l2(0.15))
        self.dropout_2 = Dropout(0.2)
        self.dense_3 = Dense(num_class, activation='softmax')

    def call(self, input_tensor, training=True):
        lstm_input = []
        for i in range(15):
            lstm_input.append(self.backbone(input_tensor[:, :, :, i, :], training=False))
        lstm_input = tf.stack(lstm_input, axis=1)
        output_lstm = self.rnn(lstm_input)
        output_lstm = tf.transpose(output_lstm, [1, 0, 2])
        dense = self.bn(output_lstm[-1], training=training)
        dense = self.dense_1(dense)
        dense = self.dropout_1(dense)
        dense = self.dense_2(dense)
        dense = self.dropout_2(dense)
        dense = self.dense_3(dense)

        return dense










