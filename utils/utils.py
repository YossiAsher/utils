import wandb

from sequences import DataGenerator


def get_datasets(project, artifact, batch_size, dim_size, input_shape):
    run = wandb.init(project=project, job_type="download", reinit=True)
    data_split_at = run.use_artifact(artifact + ":latest")
    data_split_dir = data_split_at.download()

    train_dataset = DataGenerator(task='train', path=data_split_dir, batch_size=batch_size, dim_size=dim_size,
                                  input_shape=input_shape, shuffle=False, supervised=True)
    test_dataset = DataGenerator(task='test', path=data_split_dir, batch_size=batch_size, dim_size=dim_size,
                                 input_shape=input_shape, shuffle=False, supervised=True)

    return train_dataset, test_dataset
