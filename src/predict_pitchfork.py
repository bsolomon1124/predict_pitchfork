# flake8: noqa
from collections import defaultdict
import os
import pickle

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score

from custom_scorers import wght_recall, wght_scorer  # noqa


PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

with open(os.path.join(PATH, 'data/grids.pickle'), 'rb') as file:
    grids = pickle.load(file)

with open(os.path.join(PATH, 'data/data.pickle'), 'rb') as file:
    X_train, X_test, content_train, content_test, y_train, y_test = pickle.load(file)  # noqa

# Raw probabilities (on y_train) for each grid estimator;
# we will then optimize a prediction threshold over train data.
prob = {}
for k, grid in grids.items():
    mat = X_train if k == 'x' else content_train
    prob[k] = grid.predict_proba(mat)
prob_mean = np.mean(tuple(prob.values()), axis=0)


def custom_predict(proba: np.ndarray, threshold: float):
    """Threshold on something other than 0.5, applied to multiclass case.

    Same concept/intent as upsampling.

    If p(1) & p(2) both < threshold, predict 0.
    Else predict max(p(1), p(2))
    """
    proba_ = proba[:, 1:]
    return np.where(proba_.max(axis=1) < threshold, 0,
                    proba_.argmax(axis=1) + 1)


scores = []
thresholds = np.arange(0., 0.5, 0.01)
for t in thresholds:
    new_pred = custom_predict(prob_mean, threshold=t)
    scores.append([wght_recall(y_train, new_pred),
                   f1_score(y_train, new_pred, average='weighted'),
                   precision_score(y_train, new_pred, average='weighted')])

scores = pd.DataFrame(scores, columns=['wghtd_recall', 'f1_score',
                                      'precision_score'],
                      index=thresholds)

# Use the point of intersection of recall & f1 as optimal threshold.
# This admittedly has an element of arbitrarity to it, but also visually
# represents the point at which recall starts diminishing significantly
opt = scores[(scores['f1_score'] > scores['wghtd_recall'])
           & (scores['f1_score'].shift() < scores['wghtd_recall'].shift())]\
    .index[0]

scores.iloc[5:-4].plot()
plt.title('Score metrics as function of prediction threshold')
plt.ylabel('Score')
plt.xlabel('Threshold')
plt.axvline(opt, alpha=0.3, linestyle='--', color='black')
plt.savefig(os.path.join(PATH, 'imgs/scores.png'))

test_prob = {}
for k, grid in grids.items():
    mat = X_test if k == 'x' else content_test
    test_prob[k] = grid.predict_proba(mat)
test_prob_mean = np.mean(tuple(test_prob.values()), axis=0)

test_pred = custom_predict(test_prob_mean, opt)
print(wght_recall(y_true=y_test, y_pred=test_pred))
# 0.57

# Nominal and normalized confusion matrices
cm = confusion_matrix(y_true=y_test, y_pred=test_pred)
norm_cm =  cm / cm.sum(axis=1)[:, None]  # normalized confusion matrix

cols = ['None', 'Best New Music', 'Best New Reissue']
for obj, name in zip((cm, norm_cm), ('cm', 'norm_cm')):
    pd.DataFrame(obj, columns=cols, index=cols).to_csv(
        os.path.join(PATH, 'imgs/%s.csv' % name))

# Cmapped scatterplot
# Pass colors as 2d array where final col is alpha (RGBA)
p1, p2 = test_prob_mean[:, 1:].T
rgba_colors = np.zeros((len(y_test), 4))
rgba_colors[:, -1] = np.where(y_test == 0, 0.1, 0.9)  # alphas
for i in (0, 1, 2):
    rgba_colors[:, i] = np.where(y_test == i, 1, 0)
fig, ax = plt.subplots()
ax.scatter(x=p1, y=p2, c=rgba_colors)
ax.set_xlabel('$P(1)$', color='green')
ax.set_ylabel('$P(2)$', color='blue')
ax.set_title('Test-set probabilities versus actual')
plt.savefig(os.path.join(PATH, 'imgs/scatter.png'))
