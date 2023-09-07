import multiprocessing
import pickle
import time
import copy
import numpy as np
import tensorflow as tf

from functools import partial

from matplotlib import pyplot as plt
from typing import Dict
from sklearn.model_selection import train_test_split

class GermanLettersDataSet(tf.keras.utils.Sequence):

    def __init__(self, path_csv, num_threads=3):
        self.path_csv = path_csv
        self.num_threads = num_threads

        if path_csv != None:
            print("Reading CSV file...")
            self.data_lines = self.read_lines_csv()
            print("Building data set...")
            self.n = len(self.data_lines)
            self.image_data = np.zeros((self.n, 1600), dtype=int)
            self.labels = np.empty(self.n, dtype='<U23')
            self.__build_data_set()
            self.contained_classes = []

    def __eq__(self, other):
        #if other.labels[0] != self.
        #for i in range(self.image_data):
        labels = other.labels
        labels2 = self.labels
        self.show_image_data(other.image_data[5])
        self.show_image_data(self.image_data[5])
        result = np.equal(self.image_data, other.image_data)
        result2 = result.all()
        return result2

    def split_in_test_and_train(self, number_test_samples_per_class: int):
        number_all_test_samples = number_test_samples_per_class * len(self.contained_classes)
        image_data_train = np.empty((self.n - number_all_test_samples , 1600), dtype=int)
        image_data_test = np.empty((number_all_test_samples, 1600), dtype=int)
        labels_train = np.empty(self.n - number_all_test_samples, dtype='<U23')
        labels_test = np.empty(number_all_test_samples, dtype='<U23')
        # Determine the unique classes in your dataset
        unique_classes = np.unique(self.labels)

        # Split the dataset for each class
        start_index_train = 0
        start_index_test = 0
        for class_label in unique_classes:
            # Find indices of samples in the current class
            print("Splitting class: " + str(class_label))
            indices = np.where(self.labels == class_label)[0]
            np.random.shuffle(indices)
            # Select the first n_samples_per_class indices for the test set
            test_indices = indices[:number_test_samples_per_class]
            train_indices = indices[number_test_samples_per_class:]

            # Append the data to the respective lists
            end_index_train = train_indices.shape[0] + start_index_train
            image_data_train[start_index_train:end_index_train] = self.image_data[train_indices,:]
            image_data_test[start_index_test : start_index_test + number_test_samples_per_class] = self.image_data[test_indices,:]
            labels_train[start_index_train:end_index_train] = self.labels[train_indices]
            labels_test[start_index_test : start_index_test + number_test_samples_per_class ] = self.labels[test_indices]
            start_index_test += number_test_samples_per_class
            start_index_train = end_index_train

        return image_data_train, image_data_test, labels_train, labels_test


    def read_lines_csv(self):
        training_data_file = open(self.path_csv, 'r', encoding="latin-1")
        data_lines = training_data_file.readlines()
        training_data_file.close()
        return data_lines

    def parse_one_line(self, index):
        line = self.data_lines[index].split(',')
        image = np.asarray(line[1:1601])
        image = image.astype(int)
        self.image_data[index] = image
        self.labels[index] = line[0]

    def __build_data_set(self):
        start = time.time()
        for i in range(self.n):
            self.parse_one_line(i)
            if i % 1000 == 0:
                print(i)
        print("time needed: " + str((time.time() - start)))
        unique_labels, indices = np.unique(self.labels, return_index=True)
        self.contained_classes = unique_labels

    def split_ds(self, test_size: float):
        # Use Stratified Split to maintain class distribution
        image_data_train, image_data_test, labels_train, labels_test = train_test_split(self.image_data, self.labels,
                                                                                        test_size=test_size,
                                                            stratify=self.labels, random_state=2023, shuffle=True)
        # Print the resulting class distributions
        print("Train Set Class Distribution:")
        unique_train, counts_train = np.unique(labels_train, return_counts=True)
        # Create bar plot from class distribution with matplotlib
        plt.bar(unique_train, counts_train)
        # Add labels and title
        plt.xlabel("Class Labels")
        plt.ylabel("Count")
        plt.title("Class Distribution")
        plt.show()

        for cls, count in zip(unique_train, counts_train):
            print(f"Class {cls}: {count} samples")

        print("\nTest Set Class Distribution:")
        unique_test, counts_test = np.unique(labels_test, return_counts=True)
        for cls, count in zip(unique_test, counts_test):
            print(f"Class {cls}: {count} samples")

        plt.bar(unique_test, counts_test)
        # Add labels and title
        plt.xlabel("Class Labels")
        plt.ylabel("Count")
        plt.title("Class Distribution")
        plt.show()

        return image_data_train, image_data_test, labels_train, labels_test

    def visualize_class_distribution(self, output_path=None):
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        plt.figure(figsize=(8, 25))
        plt.barh(unique_labels, counts)
        # Add labels and title
        plt.xlabel("Class Labels")
        plt.ylabel("Count")
        plt.title("Class Distribution")
        if output_path != None:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()

    def rename_ds(self, renaming_dict: Dict[str, str]) -> (np.array, np.array):
        for i in range(self.n):
            if self.labels[i] in renaming_dict.keys():
                self.labels[i] = renaming_dict[self.labels[i]]
        unique_labels, indices = np.unique(self.labels, return_index=True)
        self.contained_classes = unique_labels
        return self.image_data, self.labels

    def save_as_pickle(self, path='german_letter_ds.obj'):
        file = open(path, 'wb')
        pickle.dump(self, file)
        file.close()

    @classmethod
    def load_from_pickle(cls, path='german_letter_ds.obj'):
        file = open(path, 'rb')
        ds_instance = pickle.load(file)
        file.close()
        return ds_instance

    @classmethod
    def build_ds_from_array(cls, image_data, labels):
        ds_instance = cls(None)
        ds_instance.image_data = image_data
        ds_instance.labels = labels
        ds_instance.n = len(labels)
        ds_instance.contained_classes = np.unique(labels)
        return ds_instance

    def show_image_data(self, image_data):
        img = image_data.reshape(40, 40, 1)
        grayscale_image = np.repeat(img, 3, axis=2)
        plt.imshow(grayscale_image)
        plt.show()

    def show_all_contained_images(self):
        unique_labels, indices = np.unique(self.labels, return_index=True)
        images = self.image_data[indices]
        images = images.reshape(-1, 40, 40, 1)
        grayscale_images = np.repeat(images, 3, axis=3)
        num_images = len(images)
        fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
        print(unique_labels)
        # Display images with labels
        for i in range(num_images):
            axes[i].imshow(grayscale_images[i], cmap='gray')  # You can change the cmap as needed
            #axes[i].set_title(unique_labels[i])
            axes[i].axis('off')
        plt.show()

    def sort_ds_by_labels(self):
        sorted_indices = np.argsort(self.labels)
        # Sort the images and labels according to the unicode
        self.labels = self.labels[sorted_indices]
        self.image_data = self.image_data[sorted_indices]

    def export_to_csv(self, path: str):
        with open(path, 'w') as file:
            self.sort_ds_by_labels()
            for i in range(self.labels.shape[0]):
                export_line = self.labels[i] + ","
                data_line = ', '.join(map(str, self.image_data[i]))
                export_line += data_line
                if i % 1000 == 0 and i > 1:
                    print("exported: " + str(i))
                if i != self.labels.shape[0] - 1:
                     export_line += "\n"
                file.write(export_line)

    def concat_two_sets(self, other):
        self.image_data = np.concatenate((self.image_data, other.image_data), axis=0)
        self.labels = np.concatenate((self.labels, other.labels), axis=0)
        self.sort_ds_by_labels()
        self.contained_classes = np.unique(self.labels)
        self.n = len(self.labels)

    def calculate_mean_and_std(self):
        mean = np.mean(self.image_data)
        std = np.std(self.image_data)
        print("mean: " + str(mean) + " std: " + str(std))
        return mean, std


if __name__ == '__main__':
    #ds_all_numbers = GermanLettersDataSet('CHG_Datenbank_komplett.csv')
    #ds = GermanLettersDataSet('CHG_Datenbank_Test.csv')
    #ds.sort_ds_by_labels()
    renaming_dict = {'0': '\u0030', '1': '\u0031', '2': '\u0032', '3': '\u0033', '4': '\u0034', '5': '\u0035',
                    '6': '\u0036', '7': '\u0037', '8': '\u0038', '9': '\u0039', '12': '\u002B', '10': '\u0028',
                    '11': '\u0029', '13': '\u221A'}
    #ds.sort_ds_by_labels()
    #ds = GermanLettersDataSet.load_from_pickle("test_numbers.obj")
    #ds.rename_ds(renaming_dict)
    #ds.show_all_contained_images()
    #print(ds.contained_classes)
    #ds_all_numbers.rename_ds(renaming_dict)
    #ds_all_numbers.save_as_pickle("all_numbers.obj")
    #ds_all_numbers.export_to_csv("all_numbers.csv")
    #ds_all_numbers.sort_ds_by_labels()
    #ds_all_numbers = GermanLettersDataSet.load_from_pickle("all_numbers.obj")

    #ds.export_to_csv("all_numbers.csv")
    #renaming_dict = None
    #ds.rename_ds(renaming_dict)
    #ds = GermanLettersDataSet('/home/thomas/Dokumente/Projekte/CHG_data_sets/TestAlpha.csv')
    #ds = GermanLettersDataSet('/home/thomas/Dokumente/Projekte/CHG_data_sets/All_letters.csv')
    #ds.split_ds(0.2)
    #ds.save_as_pickle("all_letters_ds.obj")
    #print("stated with loading")
    #ds_all_letters = GermanLettersDataSet.load_from_pickle("all_letters_ds.obj")
    #print("finished with loading")
    #ds.split_ds(0.2)
    #for i in range(10):
    #   ds.show_image_point(i)
    #print(ds.contained_classes)
    #ds.show_all_contained_images()

    renaming_dict = {"Alpha": "alpha",  "Beta": "beta", "FrageZeichen": "question_mark", "GrößerZeichen": "greater_than",
                     "Integral": "integral", "KleinerZeichen": "smaller_than", "Summe": "sum", "Und": "ampersand",
                     "Unendlich": "infinity", "Phi": "phi", "Pi": "pi", "\x80": "euro", "Rundung": "tilde"}
    #ds_all_letters.rename_ds(renaming_dict)
    unicode_mapping = {'!': '\u0021', '$': '\u0024', 'A': '\u0041', 'B': '\u0042', 'C': '\u0043', 'D': '\u0044',
                       'E': '\u0045', 'F': '\u0046', 'G': '\u0047', 'H': '\u0048', 'I': '\u0049', 'J': '\u004A',
                       'K': '\u004B', 'L': '\u004C', 'M': '\u004D', 'N': '\u004E', 'O': '\u004F', 'P': '\u0050',
                       'Q': '\u0051', 'R': '\u0052', 'S': '\u0053', 'T': '\u0054', 'U': '\u0055', 'V': '\u0056',
                       'W': '\u0057', 'X': '\u0058', 'Y': '\u0059', 'Z': '\u005A', 'a': '\u0061', 'alpha': '\u03B1',
                       'ampersand': '\u0026', 'b': '\u0062', 'beta': '\u03B2', 'c': '\u0063', 'd': '\u0064',
                       'e': '\u0065', 'euro': '\u20AC', 'f': '\u0066', 'g': '\u0067', 'greater_than': '\u003E',
                       'h': '\u0068', 'i': '\u0069', 'infinity': '\u221E', 'integral': '\u222B', 'j': '\u006A',
                       'k': '\u006B', 'l': '\u006C', 'm': '\u006D', 'n': '\u006E', 'o': '\u006F', 'p': '\u0070',
                       'phi': '\u03C6', 'pi': '\u03C0', 'q': '\u0071', 'question_mark': '\u003F', 'r': '\u0072',
                       's': '\u0073', 'smaller_than': '\u003C', 'sum': '\u2211', 't': '\u0074', 'tilde': '\u007E',
                       'u': '\u0075', 'v': '\u0076', 'w': '\u0077', 'x': '\u0078', 'y': '\u0079', 'z': '\u007A',
                       'ß': '\u00DF'}
    #ds_all_letters.rename_ds(unicode_mapping)

    #print("Number different classes: " + str(len(ds.contained_classes)))
    #print(ds.contained_classes)
    #ds_all_letters.export_to_csv("all_letters_ds.csv")
    #ds_all_letters.save_as_pickle("all_letters_ds.obj")
    #ds_all_letters.concat_two_sets(ds_all_numbers)
    #ds_all_letters.save_as_pickle("ds_merged.obj")
    #ds_all_letters.export_to_csv("ds_merged.csv")
    ds_merged = GermanLettersDataSet.load_from_pickle("ds_merged.obj")
    ds_merged.visualize_class_distribution("whole_data_set_distribution.png")
    #ds_merged.visualize_class_distribution()
    print("loaded ds")
    #x_train, x_test, y_train, y_test = ds_merged.split_in_test_and_train(500)
    #ds_test = GermanLettersDataSet.build_ds_from_array(x_test, y_test)
    #ds_train = GermanLettersDataSet.build_ds_from_array(x_train, y_train)
    #ds_test.visualize_class_distribution("test_distribution.png")
    #ds_test.save_as_pickle("test_ds.obj")
    #ds_train.visualize_class_distribution("train_distribution.png")
    #ds_train.save_as_pickle("train_ds.obj")
    ds_train = GermanLettersDataSet.load_from_pickle("train_ds.obj")
    #ds_train.calculate_mean_and_std()
    #ds_test.export_to_csv("test.csv")
    #ds_train.export_to_csv("train.csv")
    print("finished")
    print("\',\'".join(ds_train.contained_classes))