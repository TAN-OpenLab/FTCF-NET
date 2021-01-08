import os
import random
import tensorflow as tf


class LoadDataset:
    def __init__(self, dataset='rlvs-std', batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size

    def transformer(self):
        '''
        generate training data and testing data
        :return:
        '''
        tf_train_filename, tf_train_labels, tf_test_filename, tf_test_labels = self.extract_filename_standard()
        train_dataset = tf.data.Dataset.from_tensor_slices((tf_train_filename, tf_train_labels))

        train_dataset = train_dataset.map(
            map_func=self.decode,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        train_dataset = train_dataset.shuffle(buffer_size=200).batch(self.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((tf_test_filename, tf_test_labels))
        test_dataset = test_dataset.map(
            map_func=self.decode,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(self.batch_size)

        return train_dataset, test_dataset

    def decode(self, filenames, label):
        res_tensor = tf.zeros(dtype=tf.float32, shape=[15, 224, 224, 3])
        index = 0
        for file in filenames:
            image_string = tf.io.read_file(file)
            image_decode = tf.image.decode_jpeg(image_string)
            image_decode = tf.image.resize(image_decode, [224, 224]) / 255.0
            image_decode = tf.expand_dims(image_decode, axis=0)
            res_tensor = tf.tensor_scatter_nd_update(res_tensor, [[index]], image_decode)
            index += 1
        res_tensor = tf.transpose(res_tensor, (1, 2, 0, 3))
        return res_tensor, label

    def obtain_file_list(self, root_path, path):
        data_filename = []
        data_labels = []
        tmplist = []
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split(' ')
                label = int(line[1])
                for i in range(1, 16, 1):
                    path = os.path.join(root_path, line[0], str(i) + ".jpg")
                    tmplist.append(path)
                data_filename.append(list(tmplist))
                data_labels.append(label)
                tmplist.clear()
        return data_filename, data_labels

    def extract_filename_standard(self):
        root_path = os.path.abspath(__file__ + "../../../" + "dataset")
        data_path = os.path.join(root_path, self.dataset)
        train_annotation_path = data_path + "/train_annotation.txt"
        test_annotation_path = data_path + "/val_annotation.txt"

        train_filename, train_labels = self.obtain_file_list(root_path, train_annotation_path)
        test_filename, test_labels = self.obtain_file_list(root_path, test_annotation_path)

        seed = 1995
        random.seed(seed)
        random.shuffle(train_filename)
        random.seed(seed)
        random.shuffle(train_labels)

        tf_train_filename = tf.constant(train_filename)
        tf_train_labels = tf.constant(train_labels)
        tf_test_filename = tf.constant(test_filename)
        tf_test_labels = tf.constant(test_labels)

        return tf_train_filename, tf_train_labels, tf_test_filename, tf_test_labels
