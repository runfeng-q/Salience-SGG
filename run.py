import os
import yaml
import argparse

from lightning import Trainer, seed_everything
from lightning.pytorch import loggers as pl_loggers
from pathlib import Path
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, early_stopping
from lightning.pytorch.strategies import DDPStrategy

from datasets import build_dataset
from util import vis_utils, visualizer
from models import build_model
from callbacks import ModelCheckpointWithArtifactLogging, ConfigSavingCallback


seed_everything(42, workers=True)

parser = argparse.ArgumentParser(description='Project')

parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file')
parser.add_argument('--command',
                    dest='command',
                    help='command',
                    default='fit')

args = parser.parse_args()


#load config
with open(args.filename, 'r') as file:
    config = yaml.safe_load(file)

model_name=args.filename.split('/')[-1].split('_')[0]

tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs',
                              name=config['logger']['experiment_name'],
                              version=config['logger']['version'])

Path(f"{tb_logger.log_dir}/groundtruth").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/prediction").mkdir(exist_ok=True, parents=True)
with open(f"{tb_logger.log_dir}/hparams.yaml", 'w', encoding='utf-8') as f:
    yaml.dump(data=config, stream=f, allow_unicode=True)

#build trainer
runner = Trainer(
                accelerator='gpu',
                devices=config['trainer']['device'],
                strategy='auto' if 'strategy' in config['trainer'] else DDPStrategy(find_unused_parameters=False),
                max_epochs=config['trainer']['epochs'],
                logger=[tb_logger],
                callbacks=[
                    LearningRateMonitor(),
                    ConfigSavingCallback(evaluator=config['data']['dataset']),
                    early_stopping.EarlyStopping(
                        monitor=config['callback']['monitor'],
                        mode=config['callback']['mode'],
                        patience=20,
                       verbose=True,
                                        ),
                    ModelCheckpointWithArtifactLogging(
                                             save_top_k=5,
                                             dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                             monitor=config['callback']['monitor'],
                                             save_last=True,
                                             mode=config['callback']['mode']
                                         )

                ],
                gradient_clip_val=config['trainer']['gradient_clip_val'],
                accumulate_grad_batches=config['trainer']['accumulate']

)

config['model']['device']=config['trainer']['device']
dataset = build_dataset(config['data'])
model = build_model(config['model'], model_name)

if args.command=='fit':
    runner.fit(model, dataset)
elif args.command=='eval':
    runner.test(model, dataset)
elif args.command=='val':
    runner.validate(model, dataset)
