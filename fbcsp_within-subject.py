from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from mne.decoding import CSP

import moabb
from moabb.datasets import Schirrmeister2017
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import FilterBankMotorImagery
from moabb.pipelines.utils import FilterBank

moabb.set_log_level("info")

# Load condig
config_file       = Path(__file__).parent /       'config.yaml'
local_config_file = Path(__file__).parent / 'local_config.yaml'
with config_file.open('r') as f:
    config = yaml.safe_load(f)
with local_config_file.open('r') as f:
    local_config = yaml.safe_load(f)

# Create Pipeline
pipelines = {}
pipelines["FBCSP+LogisticReg"] = make_pipeline(FilterBank(CSP(n_components=4)), LogisticRegression())

# Evaluation
dataset = Schirrmeister2017()
datasets = [dataset]

# Bank of 9 filters, by 4 Hz increment
paradigm = FilterBankMotorImagery(
    **config['paradigm_params']['base'],
    **config['paradigm_params']['filter_bank'],
)
evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=datasets,
    **config['evaluation_params']['base'],
    **config['evaluation_params']['within_session'],
    **local_config['evaluation_params']['base'],
)
results = evaluation.process(pipelines)
print(results)
