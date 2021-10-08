import os.path

import numpy as np
import tensorflow as tf
from svgpathtools import svg2paths, Path, CubicBezier, wsvg
import tempfile
import shutil
import cairosvg


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
        self.write_to_files(X, y, files, paths_list, index, self.epoc_path)
        return X, y

    @staticmethod
    def write_to_files(X, y, files, paths_list, epoc_index, path):
        epoc_index_path = os.path.join(path, str(epoc_index))
        print(epoc_index_path)
        np.savez_compressed(os.path.join(epoc_index_path, 'data'), X=X, y=y)
        loaded = np.load(os.path.join(epoc_index_path, 'data.npz'))
        X = loaded['X']
        print("X", X.shape)
        y = loaded['y']
        print("y", y.shape)
        for index, file in enumerate(files):
            file_path = os.path.join(epoc_index_path, str(index), file)
            wsvg(paths_list[index], filename=file_path)
            cairosvg.svg2png(url=file_path, write_to=file_path.replace('.svg', '.png'),
                             parent_width=100, parent_height=100)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

        shutil.rmtree(self.epoc_path.name)
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
                random_line, line_segment = self.__get_random_line()
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

    def __get_random_line(self):
        x1, y1 = np.random.uniform(0, 99, size=2)
        x2, y2 = np.random.uniform(0, self.line_size, size=2)
        line = Path()
        cubic_bezier = CubicBezier(complex(x1, y1), complex(x1, y1), complex(x1 + x2, y1 + y2),
                                   complex(x1 + x2, y1 + y2))
        line.append(cubic_bezier)
        return line, self.__segment_to_array(1, cubic_bezier)

    @staticmethod
    def __normalize_path_rotated(paths):
        new_paths = []
        rad = np.random.uniform(0, 360)
        for path in paths:
            new_path = path.rotated(rad)
            new_paths.append(new_path)
        return new_paths

    def __normalize_path(self, paths):
        index = 0
        segments = np.zeros(self.input_shape)
        paths = self.__normalize_path_rotated(paths)
        max_total, paths = self.__normalize_path_align(paths)
        paths = self.__normalize_path_scale(paths, max_total)
        for path in paths:
            for segment in path:
                segments[index, ] = self.__segment_to_array(0, segment)
                index += 1
        return paths, segments, index

    def __segment_to_array(self, path_index, segment):
        reverse = np.random.randint(2)
        if reverse == 0 and self.shuffle:
            values = [path_index, segment.start.real, segment.start.imag,
                      segment.control1.real, segment.control1.imag,
                      segment.control2.real, segment.control2.imag,
                      segment.end.real, segment.end.imag]
        else:
            values = [path_index, segment.end.real, segment.end.imag,
                      segment.control2.real, segment.control2.imag,
                      segment.control1.real, segment.control1.imag,
                      segment.start.real, segment.start.imag]
        return np.array(values)

    @staticmethod
    def __normalize_path_scale(paths, max_total):
        new_paths = []
        for path in paths:
            path = path.scaled(99 / max_total, 99 / max_total)
            new_paths.append(path)
        return new_paths

    @staticmethod
    def __normalize_path_align(paths):
        x_max_total = 0
        y_max_total = 0
        new_paths = []
        for path in paths:
            if len(path) > 0:
                path = path.scaled(1, -1)
                x_min, x_max, y_min, y_max = path.bbox()
                path = path.translated(complex(-x_min, -y_min))
                x_min, x_max, y_min, y_max = path.bbox()
                if x_max > x_max_total:
                    x_max_total = x_max
                if y_max > y_max_total:
                    y_max_total = y_max
                new_paths.append(path)
        max_total = max(x_max_total, y_max_total)
        return max_total, new_paths
