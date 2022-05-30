from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torch
from skorch.classifier import NeuralNetClassifier
from skorch.callbacks import Checkpoint, LRScheduler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import moabb
from moabb.datasets import Schirrmeister2017
from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import MotorImagery
from moabb.analysis import Results

from models import EEGNetv4

moabb.set_log_level("info")

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

# Create classifier
net_params = dict(
    module__n_classes=n_classes,
    module__in_chans=len(channels),
    module__input_window_samples=(t1 - t0) * resample,
)
net_params_update = [{k: v} if not isinstance(v, dict) else {f'{k}__{k2}': v2 for k2, v2 in v.items()} for k, v in
                     config['net_params'].items()]
net_params_update += [{k: v} if not isinstance(v, dict) else {f'{k}__{k2}': v2 for k2, v2 in v.items()} for k, v in
                      local_config['net_params'].items()]
for d in net_params_update:
    net_params.update(d)
print('Setting OneCycle scheduler max_lr based on lr param')
lr_scheduler = LRScheduler(
    policy=torch.optim.lr_scheduler.OneCycleLR,
    max_lr=config['net_params']['lr'],
    step_every='epoch',
    total_steps=config['net_params']['max_epochs'],
)
results_args = ['suffix', 'overwrite', 'hdf5_path', 'additional_columns']
fake_results = Results(CrossSubjectEvaluation, MotorImagery,
                       **{k: local_config['evaluation_params']['base'][k] for k in results_args if
                          k in local_config['evaluation_params']['base']})
checkpoint = Checkpoint(f_pickle='net.pickle', dirname=Path(fake_results.filepath).parent, monitor=None)
del fake_results
callbacks = [
    ('lr_scheduler', lr_scheduler),
    ('checkpoint', checkpoint),
]
net = NeuralNetClassifier(
    EEGNetv4,
    optimizer=torch.optim.AdamW,
    criterion=torch.nn.CrossEntropyLoss,
    callbacks=callbacks,
    **net_params,
)

pipelines = {}
pipelines["EEGNet-CrossSubject"] = make_pipeline(
    FunctionTransformer(func=np.float32, inverse_func=np.float64),
    net,
)

# Evaluation
dataset = Schirrmeister2017()
datasets = [dataset]

paradigm = MotorImagery(
    **config['paradigm_params']['base'],
    **config['paradigm_params']['single_band'],
)
evaluation = CrossSubjectEvaluation(
    paradigm=paradigm, datasets=datasets,
    pre_fit_function=lambda pipeline, dataset, subject: setattr(pipeline[1].callbacks[1][1], 'fn_prefix',
                                                                f'{dataset.__class__.__name__}_{subject}_{suffix}'),
    **config['evaluation_params']['base'],
    # **config['evaluation_params']['cross_subject'],
    **local_config['evaluation_params']['base'],
)
results = evaluation.process(pipelines)
print(results)
