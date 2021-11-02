import json
import tempfile
from pathlib import Path

import numpy as np
import wandb
from tensorflow.keras.callbacks import Callback


class ValLog(Callback):

    def __init__(self, datasets=None, table="predictions"):
        super().__init__()
        self.datasets = datasets
        self.table_name = table
        self.columns = ["epoch", "batch", "index", "dataset", "file", "svg", "target", "prediction"]

    def on_train_begin(self, logs=None):
        for dataset in self.datasets:
            dataset.epoc_path = tempfile.TemporaryDirectory()

    def on_epoch_end(self, epoch, logs=None):
        for dataset in self.datasets:
            self.send_results(epoch, dataset)
            dataset.epoc_path = tempfile.TemporaryDirectory()

    def send_results(self, epoch, dataset):
        predictions_table = wandb.Table(columns=self.columns)
        for batch in range(len(dataset)):
            epoc_batch_path = Path(dataset.epoc_path.name) / str(batch)
            data_path = Path(epoc_batch_path) / 'data.npz'
            loaded = np.load(str(data_path), allow_pickle=True)
            X = loaded['X']
            y = loaded['y']
            with open(str(Path(epoc_batch_path) / 'data.json'), 'r') as f:
                files = json.load(f)
            predictions = self.model.predict(X)
            for index in range(y.shape[0]):
                target = dataset.classes[y[index]]
                prediction = dataset.classes[np.argmax(predictions[index])]
                png_file = files[index].replace('svg', 'png')
                file = f"{target}-{png_file.split('/')[-1].split('.')[0]}"
                row = [epoch, batch, index, dataset.task, file, wandb.Image(png_file), target, prediction]
                predictions_table.add_data(*row)
        wandb.run.log({f"{self.table_name}_{dataset.task}": predictions_table})
