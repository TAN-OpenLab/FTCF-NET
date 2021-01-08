import os
import tensorflow as tf

from network.FTCF_Net import FTCF_Net
from utils.dataset_transformer import LoadDataset
from utils.running_gpu import running_gpu

running_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def validating(dataset='rwf-std', weight_path='./pretrained_model_weights/FTCF_rlvs_weights.h5'):
    model = FTCF_Net()
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_dataset, test_dataset = LoadDataset(dataset=dataset).transformer()

    @tf.function
    def test_step(images, labels):
        logits = model(images, training=False)
        test_acc(labels, logits)
        test_loss(loss_obj(labels, logits))

    model(tf.constant(value=0.1, shape=[1, 224, 224, 15, 3], dtype=tf.float32), training=False)
    model.load_weights(weight_path, by_name=False)
    print ("model 加载成功......")

    for images, labels in test_dataset:
        test_step(images, labels)

    tmp = 'Test Acc {}, Test loss {}'
    print(tmp.format(test_acc.result() * 100, test_loss.result()))


if __name__ == '__main__':
    validating(dataset='rlvs-std')
