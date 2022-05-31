from pathlib import Path
import argparse

import yaml

import torch
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
import moabb
from moabb.datasets import Schirrmeister2017
from moabb.paradigms import MotorImagery

from models import EEGNetv4
from lightning_data_modules import CrossSubjectDataModule

moabb.set_log_level("info")

TEST_SUBJECT = 1

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--devices', type=int, default=-1, help='the number of the GPU on which the models must be trained')
parser.add_argument('--accelerator', type=str, default=None)
args = parser.parse_args()
devices = None if (args.accelerator is None or args.accelerator == 'cpu') else args.devices if isinstance(args.devices, list) else [args.devices]

# Load condig
config_file = Path(__file__).parent / 'config.yaml'
local_config_file = Path(__file__).parent / 'local_config.yaml'
with config_file.open('r') as f:
    config = yaml.safe_load(f)
with local_config_file.open('r') as f:
    local_config = yaml.safe_load(f)
suffix = local_config['evaluation_params']['base']['suffix']
n_classes = config['paradigm_params']['base']['n_classes']
channels = config['paradigm_params']['base']['channels']
resample = config['paradigm_params']['base']['resample']
t0, t1 = Schirrmeister2017().interval

# Get classifier params
module_params = dict(
    n_classes=n_classes,
    in_chans=len(channels),
    input_window_samples=(t1 - t0) * resample,
)
module_params.update(config['net_params']['module'])

# Get data
dataset = Schirrmeister2017()
paradigm = MotorImagery(
    **config['paradigm_params']['base'],
    **config['paradigm_params']['single_band'],
)
X, labels, metadata = paradigm.get_data(dataset)
X = torch.tensor(X, dtype=torch.float32)
le = LabelEncoder()
labels_ids = torch.tensor(le.fit_transform(labels), dtype=torch.int64)

# Main:
datamodule = CrossSubjectDataModule(test_subject=TEST_SUBJECT, X=X, labels=labels_ids, metadata=metadata,
                                    dataloader_kwargs=dict(config['dataloader_params'], **local_config['dataloader_params']))
model = EEGNetv4(**module_params)
trainer = pl.Trainer(max_epochs=config['net_params']['max_epochs'], devices=devices,
                     accelerator=args.accelerator, enable_checkpointing=False,
                     auto_lr_find=True)
lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)
fig = lr_finder.plot(suggest=True)
fig.savefig('lr_finder.pdf')
fig.show()
