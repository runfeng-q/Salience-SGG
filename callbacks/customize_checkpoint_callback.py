import os


from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers


class ModelCheckpointWithArtifactLogging(ModelCheckpoint):

    def __init__(self, save_top_k, **kwargs):
        super().__init__(**kwargs)
        self.save_top_k = save_top_k

    def on_train_end(self,
                     trainer,
                     pl_module) -> None:
        if pl_module.global_rank == 0:
            return_dict = super().on_train_end(trainer, pl_module)
            temp_path = '/'.join(self.best_model_path.split('/')[:-1]) + '/best.ckpt'
            os.rename(self.best_model_path, temp_path)
            self.best_model_path = temp_path
            for logger in trainer.loggers:
                if isinstance(logger, pl_loggers.TensorBoardLogger):
                    folder_name = logger.log_dir
            for logger in trainer.loggers:
                if isinstance(logger, pl_loggers.CometLogger):
                    logger.experiment.log_asset_folder(os.path.join(folder_name, 'checkpoints'), log_file_name=True)
            return return_dict
