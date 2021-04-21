import tensorflow as tf
import matplotlib.pyplot as plt


class MNISTData:
    def __init__(self, training_data_ratio=0.9):
        self.training_data_ratio = training_data_ratio
        self.x_train = None
        self.y_train = None
        self.x_tune = None
        self.y_tune = None
        self.x_test = None
        self.y_test = None

        self.x_train_dataset = None
        self.x_tune_dataset = None
        self.x_test_dataset = None
        self.in_out_dim = None
        self.width = None
        self.height = None

    def load_data(self):
        mnist_train_data, mnist_test_data = tf.keras.datasets.mnist.load_data()

        self.x_train, self.y_train = mnist_train_data
        self.x_test, self.y_test = mnist_test_data

        # test code - begin
        # MNISTData.print_image(self.x_train[0])
        train_size = len(self.x_train)
        test_size = len(self.x_test)
        print("data size before split: train: %d, test: %d" % (train_size, test_size))

        # test code - end

        self.x_train, self.x_tune = MNISTData.split_data(data=self.x_train, split_ratio=self.training_data_ratio)
        self.y_train, self.y_tune = MNISTData.split_data(data=self.y_train, split_ratio=self.training_data_ratio)

        self.x_train = MNISTData.preprocessing_x(self.x_train)
        self.x_tune = MNISTData.preprocessing_x(self.x_tune)
        self.x_test = MNISTData.preprocessing_x(self.x_test)

        train_size = len(self.x_train)
        tune_size = len(self.x_tune)
        test_size = len(self.x_test)

        # test code
        # MNISTData.print_image(self.x_train[0])
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        w_by_h = self.x_train.shape[1] * self.x_train.shape[2]
        self.x_train = self.x_train.reshape(train_size, w_by_h).astype("float32")
        self.x_train_dataset = tf.data.Dataset.from_tensor_slices(self.x_train)
        self.x_tune = self.x_tune.reshape(tune_size, w_by_h).astype("float32")
        self.x_tune_dataset = tf.data.Dataset.from_tensor_slices(self.x_tune)
        self.x_test = self.x_test.reshape(test_size, w_by_h).astype("float32")
        self.x_test_dataset = tf.data.Dataset.from_tensor_slices(self.x_test)
        self.in_out_dim = w_by_h

        print("data size after split: train: %d, tune: %d, test: %d" % (train_size, tune_size, test_size))

    @staticmethod
    def split_data(data, split_ratio):
        # data must be list
        # split_ratio is portion of the first data block
        data_size = len(data)
        split_idx = int(data_size * split_ratio)
        data_split_1 = data[0:split_idx]
        data_split_2 = data[split_idx:]
        return data_split_1, data_split_2

    @staticmethod
    def preprocessing_x(x_data):
        x_data = x_data/float(255.0)
        # x_data = [0.0 if x < 0.5 else 1.0 for x in x_data] - impossible to apply to each element in vector
        return x_data

    @staticmethod
    def print_image(img_data):
        plt.figure()
        plt.imshow(img_data)
        plt.colorbar()
        plt.grid(False)
        plt.show()

    @staticmethod
    def print_56_pair_images(img_data_list1, img_data_list2, label_list):
        num_row = 7
        num_col = 16
        num_pairs = num_row * num_col
        plt.figure(figsize=(10, 8))
        plt.title("Digit pairs")
        num_images = img_data_list1.shape[0]

        if num_images > num_pairs:
            num_images = num_pairs

        for i in range(num_images):
            plt.subplot(num_row, num_col, 2*i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img_data_list1[i], cmap=plt.cm.binary)
            plt.xlabel(label_list[i])

        for i in range(num_images):
            plt.subplot(num_row, num_col, 2*(i+1))
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img_data_list2[i], cmap=plt.cm.binary)
            plt.xlabel(label_list[i])

        plt.show()

    @staticmethod
    def print_10_images(img_data_list, label_list):
        num_row = 2
        num_col = 5
        num_prints = num_row * num_col
        plt.figure(figsize=(10, 8))
        plt.title("Digit pairs")
        num_images = img_data_list.shape[0]

        if num_prints > num_images:
            num_prints = num_images

        for i in range(num_prints):
            plt.subplot(num_row, num_col, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img_data_list[i], cmap=plt.cm.binary)
            plt.xlabel(label_list[i])

        plt.show()
