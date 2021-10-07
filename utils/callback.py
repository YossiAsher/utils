from tensorflow.keras.callbacks import Callback
import numpy as np
import wandb

TABLE_NAME = "predictions"


class ValLog(Callback):
    """ Custom callback to log validation images
    at the end of each training epoch"""

    def __init__(self, dataset=None):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        for X, y in self.dataset.epoc_data:
            val_preds = self.model.predict(X)

        # log validation predictions alongside the run
        columns = ["id", "image", "guess", "truth"]

        predictions_table = wandb.Table(columns=columns)

        # log image, predicted and actual labels, and all scores
        for filepath, img, top_guess, scores, truth in zip(self.generator.filenames,
                                                           val_data,
                                                           max_preds,
                                                           val_preds,
                                                           true_ids):
            img_id = filepath.split('/')[-1].split(".")[0]
            row = [img_id, wandb.Image(img),
                   self.flat_class_names[top_guess], self.flat_class_names[truth]]
            for s in scores.tolist():
                row.append(np.round(s, 4))
            predictions_table.add_data(*row)
        wandb.run.log({TABLE_NAME: predictions_table})
