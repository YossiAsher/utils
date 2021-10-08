import numpy as np
import os
import cairosvg
from svgpathtools import Path, wsvg, Line, CubicBezier


def segment_to_array(path_index, segment, shuffle):
    reverse = np.random.randint(2)
    if reverse == 0 and shuffle:
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


def get_random_line(line_size, shuffle):
    x1, y1 = np.random.uniform(0, 99, size=2)
    x2, y2 = np.random.uniform(0, line_size, size=2)
    line = Path()
    cubic_bezier = CubicBezier(complex(x1, y1), complex(x1, y1), complex(x1 + x2, y1 + y2),
                               complex(x1 + x2, y1 + y2))
    line.append(cubic_bezier)
    return line, segment_to_array(1, cubic_bezier, shuffle)


def write_to_files(X, y, files, paths_list, epoc_index, path):
    epoc_index_path = os.path.join(path, str(epoc_index))
    os.makedirs(epoc_index_path, exist_ok=True)
    np.savez_compressed(os.path.join(epoc_index_path, 'data'), X=X, y=y)
    for index, file in enumerate(files):
        file_path = os.path.join(epoc_index_path, str(index), file)
        svg_paths = paths_list[index]
        if len(svg_paths) == 0:
            svg_paths = [Path(Line(200 + 300j, 250 + 350j))]
        wsvg(svg_paths, filename=file_path)
        cairosvg.svg2png(url=file_path, write_to=file_path.replace('.svg', '.png'),
                         parent_width=100, parent_height=100)
