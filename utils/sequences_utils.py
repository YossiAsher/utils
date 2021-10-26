import json
import os

import numpy as np
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
