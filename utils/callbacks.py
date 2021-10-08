import os.path

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
        columns = ["epoch", "dataset", "file", "target", "prediction"]
        predictions_table = wandb.Table(columns=columns)

        for i in range(10):
            path = os.path.join(self.dataset.epoc_path.name, str(i), 'data.npz')
            loaded = np.load(path)
            X = loaded['X']
            y = loaded['y']
            val_preds = self.model.predict(X)
            for index, x in enumerate(X):
                target = self.dataset.classes[y[index]]
                prediction = self.dataset.classes[np.argmax(val_preds[index])]
                row = [epoch, self.dataset.name, "files[index]", target, prediction]
                predictions_table.add_data(*row)
        self.run.log({self.table_name: predictions_table})
        self.dataset.clean_epoc_path()
