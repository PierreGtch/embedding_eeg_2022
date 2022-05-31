from pathlib import Path
import shutil
import argparse
from joblib import Parallel, delayed

import yaml

import torch
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
import moabb
from moabb.datasets import Schirrmeister2017
from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import MotorImagery
from moabb.analysis import Results

from models import EEGNetv4
from lightning_data_modules import CrossSubjectDataModule

moabb.set_log_level("info")

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=int, nargs='+')
parser.add_argument('--devices', type=int, default=None, help='the number of the GPU on which the models must be trained')
parser.add_argument('--accelerator', type=str, default=None)
parser.add_argument('--n_jobs', type=int, default=1)
parser.add_argument('--overwrite_checkpoints', default=False, action='store_true')
args = parser.parse_args()
devices = None if (args.accelerator is None or args.accelerator == 'cpu') else \
    [args.devices] if isinstance(args.devices, int) else args.devices

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

# Prepare checkpoint directories
results_param_names = ['suffix', 'overwrite', 'hdf5_path', 'additional_columns']
results_params = {k: local_config['evaluation_params']['base'][k] for k in results_param_names if
                  k in local_config['evaluation_params']['base']}
fake_results = Results(CrossSubjectEvaluation, MotorImagery, **results_params)
checkpoints_root_dir = Path(fake_results.filepath).parent
del fake_results
checkpoint_dir_list = []
for subject in args.subjects:
    path = checkpoints_root_dir / (str(subject) + results_params['suffix'])
    if path.exists():
        if args.overwrite_checkpoints:
            print(f'removing pre-existing checkpoint directory {path}')
            shutil.rmtree(path)
        else:
            raise ValueError(f'Checkpoint directory {path} already exists')
    path.mkdir(parents=True)
    checkpoint_dir_list.append(str(path))

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


# Define training loop:
def main(subject, ckpt_path):
    print(subject, ckpt_path)
    datamodule = CrossSubjectDataModule(test_subject=subject, X=X, labels=labels_ids, metadata=metadata,
                                        dataloader_kwargs=dict(config['dataloader_params'], **local_config['dataloader_params']))
    model = EEGNetv4(**module_params)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=ckpt_path)
    trainer = pl.Trainer(max_epochs=config['net_params']['max_epochs'], devices=devices,
                         accelerator=args.accelerator, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=datamodule)
    return trainer.test(model, datamodule=datamodule, verbose=True)


if args.n_jobs > 1:
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(main)(subject, ckpt_path) for subject, ckpt_path in zip(args.subjects, checkpoint_dir_list)
    )
else:
    results = [main(subject, ckpt_path) for subject, ckpt_path in zip(args.subjects, checkpoint_dir_list)]

print(results)
