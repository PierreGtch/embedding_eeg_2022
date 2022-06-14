from pathlib import Path

import yaml

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import moabb
from moabb.datasets import Schirrmeister2017
from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import MotorImagery
from moabb.analysis import Results

from models import EEGNetv4
from skorch_frozen import FrozenNeuralNetClassifier

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

# Dataset
dataset = Schirrmeister2017()
datasets = [dataset]

paradigm = MotorImagery(
    **config['paradigm_params']['base'],
    **config['paradigm_params']['single_band'],
)

# Prepare checkpoint directories
results_param_names = ['suffix', 'overwrite', 'hdf5_path', 'additional_columns']
results_params = {k: local_config['evaluation_params']['base'][k] for k in results_param_names if
                  k in local_config['evaluation_params']['base']}
fake_results = Results(CrossSubjectEvaluation, MotorImagery, **results_params)
checkpoints_root_dir = Path(fake_results.filepath).parent
del fake_results
checkpoints_dict = {}
for subject in dataset.subject_list:
    path = checkpoints_root_dir / (str(subject) + results_params['suffix'])
    files = list(path.glob('*.ckpt'))
    if len(files) != 1:
        raise ValueError(f'Multiple or no checkpoint file(s) present at {path}')
    checkpoints_dict[subject] = str(files[0])

# Create pipeline
pipelines = {}
pipelines["EEGNet-CrossSubject"] = make_pipeline(
    FunctionTransformer(func=np.float32, inverse_func=np.float64),
    FrozenNeuralNetClassifier(EEGNetv4.load_from_checkpoint(str(list(checkpoints_dict.values())[0]))),
)


def pre_fit_function(pipeline, dataset, subject):
    path = checkpoints_dict[subject]
    print(f'Loading checkpoint for subject {subject} from {path}')
    pipeline[1].initialize().module.load_state_dict(EEGNetv4.load_from_checkpoint(path).state_dict())


# Evaluation
evaluation = CrossSubjectEvaluation(
    paradigm=paradigm, datasets=datasets,
    pre_fit_function=pre_fit_function,
    **config['evaluation_params']['base'],
    # **config['evaluation_params']['cross_subject'],
    **local_config['evaluation_params']['base'],
)
results = evaluation.process(pipelines)
print(results)
