from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection._univariate_selection import _clean_nans
from mne.decoding import CSP
from mne import set_log_level as mne_set_log_level

from moabb import set_log_level as moabb_set_log_level
from moabb.datasets import Schirrmeister2017
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import FilterBankMotorImagery
from moabb.pipelines.utils import FilterBank

mne_set_log_level(0)
moabb_set_log_level("info")

# Load condig
config_file = Path(__file__).parent / 'config.yaml'
local_config_file = Path(__file__).parent / 'local_config.yaml'
with config_file.open('r') as f:
    config = yaml.safe_load(f)
with local_config_file.open('r') as f:
    local_config = yaml.safe_load(f)


# Create Pipeline
class SelectKBestCSP(SelectKBest):
    '''
    must be used with only two classes and CSP(n_components=n_csp_components, component_order='alternate')
    '''

    def __init__(self, score_func=None, *, n_csp_components=None, k=10):
        if n_csp_components % 2 != 0:
            raise ValueError()
        super().__init__(score_func=score_func, k=k)
        self.n_csp_components = n_csp_components

    def _get_support_mask(self):
        check_is_fitted(self)

        if self.k == "all":
            return np.ones(self.scores_.shape, dtype=bool)
        elif self.k == 0:
            return np.zeros(self.scores_.shape, dtype=bool)
        else:
            scores = _clean_nans(self.scores_)
            mask = np.zeros(scores.shape, dtype=bool)
            for i in range(self.k):
                if np.all(mask):
                    break
                imax = np.arange(len(mask))[~mask][np.argmax(scores[~mask])]
                other = 1 if imax % 2 == 0 else -1
                mask[imax] = 1
                # select the matching component:
                assert mask[imax + other] == False
                mask[imax + other] = 1
            return mask


pipelines = {}
pipelines["FBCSP+LogisticReg"] = make_pipeline(FilterBank(CSP(n_components=4)), LogisticRegression())
pipelines["OVR FBCSP+MIBIF4+LogisticReg"] = OneVsRestClassifier(
    make_pipeline(FilterBank(CSP(n_components=4, component_order='alternate')),
                  SelectKBestCSP(mutual_info_classif, n_csp_components=4, k=4), LogisticRegression()), n_jobs=4)

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
