import json
import os.path
import tempfile

import numpy as np
import wandb
from tensorflow.keras.callbacks import Callback


class ValLog(Callback):

    def __init__(self, datasets=None, table="predictions", project=None, run=None):
        super().__init__()
        self.project = project
        self.run = run
        self.datasets = datasets
        self.table_name = table
        self.columns = ["epoch", "batch", "index", "dataset", "location", "file", "svg", "target", "prediction"]
        self.run = wandb.init(project=self.project, job_type="inference", name=self.run)

    def on_epoch_end(self, epoch, logs=None):
        for dataset in self.datasets:
            self.send_results(epoch, dataset)

    def on_epoch_begin(self, epoch, logs=None):
        print(epoch, logs)
        for dataset in self.datasets:
            print(dataset.task)
            dataset.epoc_path = tempfile.TemporaryDirectory()

    def send_results(self, epoch, dataset):
        print(epoch, dataset.task)
        predictions_table = wandb.Table(columns=self.columns)
        for batch in range(len(dataset)):
            epoc_batch_path = os.path.join(dataset.epoc_path.name, str(batch))
            data_path = os.path.join(epoc_batch_path, 'data.npz')
            loaded = np.load(data_path, allow_pickle=True)
            X = loaded['X']
            y = loaded['y']
            with open(os.path.join(epoc_batch_path, 'data.json'), 'r') as f:
                files = json.load(f)
            predictions = self.model.predict(X)
            for index in range(y.shape[0]):
                target = dataset.classes[y[index]]
                prediction = dataset.classes[np.argmax(predictions[index])]
                png_file = files[index].replace('svg', 'png')
                file = png_file.split('/')[-1].split('.')[0]
                row = [epoch, batch, index, dataset.task, dataset.epoc_path.name,
                       file, wandb.Image(png_file), target, prediction]
                predictions_table.add_data(*row)
        self.run.log({f"{self.table_name}_{dataset.task}": predictions_table})
