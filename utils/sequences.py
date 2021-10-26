import glob
import os
import os.path

import numpy as np
import tensorflow as tf
from utils.sequences_utils import init_data, write_to_files, get_random_line, \
    normalize_path_rotated, normalize_path_align, segment_to_array, normalize_path_scale


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, task, path, batch_size, dim_size, input_shape, line_size=70,
                 supervised=False, shuffle=True, debug=False):
        self.input_shape = input_shape
        self.supervised = supervised
        self.task = task
        self.debug = debug
        self.path = f"{path}/svg/{task}"
        self.files = set(glob.glob(f"{self.path}/**/*.svg", recursive=True))
        self.classes = os.listdir(self.path)
        self.classes.sort()
        self.shuffle = shuffle
        self.batch_size = batch_size
        if self.batch_size == -1:
            self.batch_size = len(self.files)
        self.dim_size = dim_size
        self.line_size = line_size
        self.data = init_data(self.files)
        self.indexes = np.arange(len(self.data))
        self.epoc_path = None
        print("data: ", len(self.data))
        print("classes: ", self.classes)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        data_temp = [self.data[k] for k in indexes]

        # Generate data
        X, y, files = self.__data_generation(data_temp)
        if self.epoc_path:
            data_path = os.path.join(self.epoc_path.name, str(index))
            if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
                write_to_files(X, y, files, data_path)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.zeros((self.batch_size, *self.input_shape))
        y = np.zeros(self.batch_size, dtype=int)
        files = []

        # Generate data
        for i, (paths, file) in enumerate(data_temp):
            paths, segments, index = self.__normalize_path(paths)
            out = 0
            if self.supervised:
                class_name = file.split('/')[-2]
                out = self.classes.index(class_name)
            else:
                random_line, line_segment = get_random_line(self.line_size)
                segments[index] = line_segment
                for path in paths:
                    if len(path.intersect(random_line)) > 0:
                        out = 1

            np.random.shuffle(segments)
            X[i, ] = segments
            y[i] = out
            files.append(file)
        return X, y, files

    def __normalize_path(self, paths):
        index = 0
        segments = np.zeros(self.input_shape)
        paths = normalize_path_rotated(paths)
        max_total, paths = normalize_path_align(paths)
        paths = normalize_path_scale(paths, max_total)
        for path in paths:
            for segment in path:
                segments[index, ] = segment_to_array(0, segment)
                index += 1
        return paths, segments, index
