import os.path
import shutil
import tempfile

import tensorflow as tf
from svgpathtools import svg2paths

from helper import *


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, name, files, batch_size, dim_size, input_shape, line_size=70,
                 supervised=False, shuffle=True, debug=False):
        self.input_shape = input_shape
        self.supervised = supervised
        self.files = files
        self.name = name
        self.classes = list(set([f.split('/')[-2] for f in self.files])) if supervised else [0, 1]
        print("files: ", len(self.files))
        self.shuffle = shuffle
        self.debug = debug
        self.batch_size = batch_size
        if self.batch_size == -1:
            self.batch_size = len(self.files)
        self.dim_size = dim_size
        self.line_size = line_size
        self.data = self.__init_data(self.files)
        self.indexes = np.arange(len(self.data))
        self.epoc_path = tempfile.TemporaryDirectory()
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        data_temp = [self.data[k] for k in indexes]

        # Generate data
        X, y, files, paths_list = self.__data_generation(data_temp)
        self.write_to_files(X, y, files, paths_list, index, self.epoc_path.name)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def clean_epoc_path(self):
        if self.epoc_path and os.path.exists(self.epoc_path.name):
            shutil.rmtree(self.epoc_path.name, ignore_errors=True)
        self.epoc_path = tempfile.TemporaryDirectory()

    @staticmethod
    def __init_data(files):
        data = []
        for file in files:
            paths, attributes = svg2paths(file)
            data.append((paths, file))
        return data

    def __data_generation(self, data_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.zeros((self.batch_size, *self.input_shape))
        y = np.zeros(self.batch_size, dtype=int)
        files = []
        paths_list = []

        # Generate data
        for i, (paths, file) in enumerate(data_temp):
            # Store sample
            paths, segments, index = self.__normalize_path(paths)
            random_line = None
            out = 0
            if self.supervised:
                class_name = file.split('/')[-2]
                out = self.classes.index(class_name)
            else:
                random_line, line_segment = get_random_line(self.line_size, self.shuffle)
                segments[index] = line_segment
                for path in paths:
                    if len(path.intersect(random_line)) > 0:
                        out = 1

            if self.debug:
                if random_line:
                    paths.append(random_line)

            if self.shuffle:
                np.random.shuffle(segments)
            X[i, ] = segments
            y[i] = out
            files.append(file)
            paths_list.append(paths)
        return X, y, files, paths_list

    def __normalize_path(self, paths):
        index = 0
        segments = np.zeros(self.input_shape)
        paths = normalize_path_rotated(paths)
        max_total, paths = normalize_path_align(paths)
        paths = normalize_path_scale(paths, max_total)
        for path in paths:
            for segment in path:
                segments[index, ] = segment_to_array(0, segment, self.shuffle)
                index += 1
        return paths, segments, index
