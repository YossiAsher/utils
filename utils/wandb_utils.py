import wandb
import glob
import os

import cairosvg
import numpy as np

from utils.sequences import DataGenerator


def get_datasets(project, artifact, batch_size, dim_size, input_shape):
    run = wandb.init(project=project, job_type="download", reinit=True)
    data_split_at = run.use_artifact(artifact + ":latest")
    data_split_dir = data_split_at.download()

    train_dataset = DataGenerator(task='train', path=data_split_dir, batch_size=batch_size, dim_size=dim_size,
                                  input_shape=input_shape, shuffle=False, supervised=True)
    test_dataset = DataGenerator(task='test', path=data_split_dir, batch_size=batch_size, dim_size=dim_size,
                                 input_shape=input_shape, shuffle=False, supervised=True)
    unsupervised_dataset = DataGenerator(task='unsupervised', path=data_split_dir, batch_size=batch_size,
                                         dim_size=dim_size,
                                         input_shape=input_shape, shuffle=False, supervised=False)

    return train_dataset, test_dataset, unsupervised_dataset


def create_raw_data(project, artifact, path):
    raw_files = glob.glob(f"{path}/**/*.svg", recursive=True)
    run = wandb.init(project=project, job_type="upload", reinit=True)
    raw_data_at = wandb.Artifact(artifact, type="raw_data")

    for file_path in raw_files:
        label = file_path.split('/')[-2]
        file = file_path.split('/')[-1]
        raw_data_at.add_file(file_path, name=label + "/" + file)

    run.log_artifact(raw_data_at)


def split_data(project, raw_artifact, split_artifact):
    run = wandb.init(project=project, job_type="data_split", reinit=True)
    data_at = run.use_artifact(raw_artifact + ":latest")
    data_dir = data_at.download()
    data_split_at = wandb.Artifact(split_artifact, type="balanced_data")

    labels = os.listdir(data_dir)
    for label in labels:
        images_per_label = os.listdir(os.path.join(data_dir, label))
        for img_file in images_per_label:
            img_id = img_file.split('.')[0]
            full_path = os.path.join(data_dir, label, img_file)
            split = "test" if np.random.rand() > 0.8 else "train"
            data_split_at.add_file(full_path, name=f"svg/{split}/{label}/{img_id}.svg")
            data_split_at.add_file(full_path, name=f"svg/unsupervised/{label}/{img_id}.svg")
            png_full_path = full_path.replace('.svg', '.png')
            cairosvg.svg2png(url=full_path, write_to=png_full_path, parent_width=100, parent_height=100)
            data_split_at.add_file(png_full_path, name=f"png/{split}/{label}/{img_id}.png")
            data_split_at.add_file(png_full_path, name=f"png/unsupervised/{label}/{img_id}.png")

    run.log_artifact(data_split_at)
