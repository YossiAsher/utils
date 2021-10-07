import wandb
from tensorflow.keras.callbacks import Callback

TABLE_NAME = "predictions"


class ValLog(Callback):
    """ Custom callback to log validation images
    at the end of each training epoch"""

    def __init__(self, dataset=None):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        columns = ["file", "prediction", "target"]

        predictions_table = wandb.Table(columns=columns)

        # log image, predicted and actual labels, and all scores
        for X, y, files, paths_list in self.dataset.epoc_data:
            val_preds = self.model.predict(X)
            for index, x in enumerate(X):
                row = [files[index], val_preds[index], y[index]]
                predictions_table.add_data(*row)
        wandb.run.log({TABLE_NAME: predictions_table})
