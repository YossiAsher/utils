import os.path
import glob

import wandb
import numpy as np
from tensorflow.keras.callbacks import Callback


class ValLog(Callback):

    def __init__(self, datasets=None, table="predictions", project="svg-attention6", run=""):
        super().__init__()
        self.datasets = datasets
        self.table_name = table
        self.run = wandb.init(project=project, job_type="inference", name=run)
        self.columns = ["epoch", "batch", "index", "dataset", "location", "file", "svg", "target", "prediction"]

    def on_epoch_end(self, epoch, logs=None):
        for dataset in self.datasets:
            self.ex_dataset(epoch, dataset)

    def ex_dataset(self, epoch, dataset):
        predictions_table = wandb.Table(columns=self.columns)
        for batch in range(len(dataset)):
            epoc_path_index = os.path.join(dataset.epoc_path.name, str(batch))
            data_path = os.path.join(epoc_path_index, 'data.npz')
            loaded = np.load(data_path, allow_pickle=True)
            X = loaded['X']
            y = loaded['y']
            predictions = self.model.predict(X)
            for index in range(y.shape[0]):
                target = dataset.classes[y[index]]
                prediction = dataset.classes[np.argmax(predictions[index])]
                png_file = glob.glob(f'{epoc_path_index}/{index}/**/*.png', recursive=True)[0]
                file = png_file.split('/')[-1]
                row = [epoch, batch, index, dataset.task, dataset.epoc_path.name,
                       file, wandb.Image(png_file), target, prediction]
                predictions_table.add_data(*row)
        self.run.log({f"{self.table_name}_{dataset.task}": predictions_table})
        # self.dataset.clean_epoc_path()
