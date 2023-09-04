import tensorflow as tf
import numpy as np

class GermanCharacaterRecognitionDS(tf.keras.utils.Sequence):

    def __init__(self, path_csv,
                 batch_size,
                 input_size=(40, 40, 1),
                 shuffle=True):
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.path_csv = path_csv
        self.data_lines = self.__read_lines_csv(self.path_csv)
        self.n = len(self.data_lines)

    def __read_lines_csv(self):
        training_data_file = open(self.path_csv, 'r', encoding="latin-1")
        data_lines = training_data_file.readlines()
        training_data_file.close()
        return data_lines

    def __parse_one_line(self, index):
        line = self.data_lines[index].split(',')
        image = np.asarray(line[1:1601])
        image = image.astype(int)
        self.image_data[index] = image
        self.labels[index] = line[0]

    def on_epoch_end(self):
        #if self.shuffle:
        #    self.df = self.df.sample(frac=1).reset_index(drop=True)
        pass

    def __get_input(self, path, bbox, target_size):
        xmin, ymin, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = image_arr[ymin:ymin + h, xmin:xmin + w]
        image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()

        return image_arr / 255.

    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)

    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches[self.X_col['path']]
        bbox_batch = batches[self.X_col['bbox']]

        name_batch = batches[self.y_col['name']]
        type_batch = batches[self.y_col['type']]

        X_batch = np.asarray([self.__get_input(x, y, self.input_size) for x, y in zip(path_batch, bbox_batch)])

        y0_batch = np.asarray([self.__get_output(y, self.n_name) for y in name_batch])
        y1_batch = np.asarray([self.__get_output(y, self.n_type) for y in type_batch])

        return X_batch, tuple([y0_batch, y1_batch])

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size