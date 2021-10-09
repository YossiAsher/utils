import os.path
import glob

import wandb
import numpy as np
from tensorflow.keras.callbacks import Callback


class ValLog(Callback):

    def __init__(self, dataset=None, table="predictions", project="svg-attention6", run=""):
        super().__init__()
        self.dataset = dataset
        self.table_name = table
        self.run = wandb.init(project=project, job_type="inference", name=run)

    def on_epoch_end(self, epoch, logs=None):
        columns = ["epoch", "dataset", "file", "svg", "target", "prediction"]
        predictions_table = wandb.Table(columns=columns)

        for i in range(len(self.dataset)):
            epoc_path_index = os.path.join(self.dataset.epoc_path.name, str(i))
            data_path = os.path.join(epoc_path_index, 'data.npz')
            loaded = np.load(data_path, allow_pickle=True)
            X = loaded['X']
            y = loaded['y']
            predictions = self.model.predict(X)
            for index, x in enumerate(X):
                target = self.dataset.classes[y[index]]
                prediction = self.dataset.classes[np.argmax(predictions[index])]
                png_file = glob.glob(f'{epoc_path_index}/{index}/**/*.png', recursive=True)[0]
                file = png_file[png_file.index(png_file.split('/')[-2]):]
                row = [epoch, self.dataset.name, file, wandb.Image(png_file), target, prediction]
                predictions_table.add_data(*row)
        self.run.log({self.table_name: predictions_table})
        # self.dataset.clean_epoc_path()
