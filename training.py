import sys
import math
import tensorflow as tf

from network.FTCF_Net import FTCF_Net
from experiments_net import C3D, Conv_LSTM, DenseNet_3D, InceptionNet_3D, Inflated_3D, ResNet_3D
from utils.dataset_transformer import LoadDataset
from utils.running_gpu import running_gpu

running_gpu()


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def build_model(model_name):
    def default():
        print ("No such Model!, Please choose model from C3D, Conv_LSTM, DenseNet_3D, InceptionNet_3D, Inflated_3D, \
        ResNet_3D, FTCF_Net")
    switch = {
        'C3D': C3D.C3D_Model(),
        'Conv_LSTM': Conv_LSTM.Conv_LSTM_Model(),
        'DenseNet_3D': DenseNet_3D.DenseNet_3D_Model(),
        'InceptionNet_3D': InceptionNet_3D.InceptionNet_3D_Model(),
        'Inflated_3D': Inflated_3D.Inflated_3D_Model(),
        'ResNet_3D': ResNet_3D.ResNet_3D_Model(),
        'FTCF_Net': FTCF_Net()
    }
    return switch.get(model_name, default)


def training(epochs=50, batch_size=32, model_name='FTCF_Net', dataset='rlvs-std', lr=1e-3, mom=0.9):
    if dataset == 'rlvs-std':
        train_length = 1600
        test_length = 400
    elif dataset == 'hockey-std':
        train_length = 800
        test_length = 200
    elif dataset == 'peliculas-std':
        train_length = 161
        test_length = 40
    elif dataset == 'violentflow-std':
        train_length = 196
        test_length = 50

    model = build_model(model_name)

    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=mom, decay=1e-5)

    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_dataset, test_dataset = LoadDataset(batch_size=batch_size, dataset=dataset).transformer()

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = loss_obj(labels, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_acc(labels, logits)
        train_loss(loss)

    @tf.function
    def test_step(images, labels):
        logits = model(images, training=False)
        test_acc(labels, logits)
        test_loss(loss_obj(labels, logits))

    for epoch in range(epochs):
        train_acc.reset_states()
        train_loss.reset_states()
        test_acc.reset_states()
        index = 0
        for images, labels in train_dataset:

            train_step(images, labels)
            index += len(labels)
            view_bar("training ", index, train_length)

        print("")
        index = 0
        for images, labels in test_dataset:
            test_step(images, labels)
            index += len(labels)
            view_bar("testing", index, test_length)

        print("")
        tmp = 'Epoch {}, Acc {}, Train loss {},Test Acc {}, Test loss {}'
        print(tmp.format(epoch + 1,
                         train_acc.result() * 100,
                         train_loss.result(),
                         test_acc.result() * 100,
                         test_loss.result()))


if __name__ == '__main__':
    training(epochs=100, batch_size=16, model_name='FTCF_Net', dataset='rlvs-std')
