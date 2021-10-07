import wandb
import numpy as np
from tensorflow.keras.callbacks import Callback


class ValLog(Callback):

    def __init__(self, dataset=None, table_name="predictions", project_name="svg-attention6", run_name=""):
        super().__init__()
        self.dataset = dataset
        self.table_name = table_name
        self.run = wandb.init(project=project_name, job_type="inference", name=run_name)

    def on_epoch_end(self, epoch, logs=None):
        columns = ["file", "target", "prediction"]
        predictions_table = wandb.Table(columns=columns)
        
        for X, y, files, paths_list in self.dataset.epoc_data.values():
            val_preds = self.model.predict(X)
            for index, x in enumerate(X):
                target = self.dataset.classes[y[index]]
                prediction = self.dataset.classes[np.argmax(val_preds[index])]
                row = [files[index], target, prediction]
                predictions_table.add_data(*row)
        self.run.log({self.table_name: predictions_table})
