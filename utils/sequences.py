import glob
import json
import os
import os.path
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from svgpathtools import Path, CubicBezier, svg2paths


def init_data(files):
    data = []
    for file in files:
        paths, attributes = svg2paths(file)
        if len(paths) > 0:
            data.append((paths, file))
    return data


def segment_to_array(path_index, segment):
    reverse = np.random.randint(2)
    if reverse == 0:
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


def normalize_path_scale(paths, max_total):
    new_paths = []
    for path in paths:
        path = path.scaled(99 / max_total, 99 / max_total)
        new_paths.append(path)
    return new_paths


def normalize_path_align(paths):
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


def normalize_path_rotated(paths):
    new_paths = []
    rad = np.random.uniform(0, 360)
    for path in paths:
        new_path = path.rotated(rad)
        new_paths.append(new_path)
    return new_paths


def get_random_line(line_size):
    x1, y1 = np.random.uniform(0, 99, size=2)
    x2, y2 = np.random.uniform(0, line_size, size=2)
    line = Path()
    cubic_bezier = CubicBezier(complex(x1, y1), complex(x1, y1), complex(x1 + x2, y1 + y2),
                               complex(x1 + x2, y1 + y2))
    line.append(cubic_bezier)
    return line, segment_to_array(1, cubic_bezier)


def write_to_files(X, y, files, batch_path):
    os.makedirs(batch_path, exist_ok=True)
    np.savez_compressed(os.path.join(batch_path, 'data'), X=X, y=y)
    with open(os.path.join(batch_path, 'data.json'), 'w') as f:
        json.dump(files, f)


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, task, path, batch_size, dim_size, input_shape, line_size=70,
                 supervised=False, shuffle=True, debug=False):
        self.input_shape = input_shape
        self.supervised = supervised
        self.task = task
        self.debug = debug
        self.path = f"{path}/{task}"
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
        self.epoc_path = tempfile.TemporaryDirectory()
        self.last_epoc_path = None
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

        print('__getitem__', index, self.task)
        print(self.epoc_path.name)
        data_path = os.path.join(self.epoc_path.name, str(index))
        print(os.listdir(data_path))
        # Generate data
        X, y, files = self.__data_generation(data_temp)
        if len(os.listdir(data_path)) == 0:
            write_to_files(X, y, files, data_path)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

        print(self.epoc_path.name)
        # print(self.last_epoc_path.name)
        # if self.last_epoc_path and os.path.exists(self.last_epoc_path.name):
        #     print("rm: ", self.last_epoc_path.name)
        #     shutil.rmtree(self.last_epoc_path.name, ignore_errors=True)
        # self.last_epoc_path = self.epoc_path
        self.epoc_path = tempfile.TemporaryDirectory()
        print(self.epoc_path.name)
        # print(self.last_epoc_path.name)

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
            X[i,] = segments
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
                segments[index,] = segment_to_array(0, segment)
                index += 1
        return paths, segments, index
