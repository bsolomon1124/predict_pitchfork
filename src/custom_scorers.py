"""Custom scorers for multilabel classification."""

import numpy as np
from sklearn.metrics import make_scorer, confusion_matrix


# NOTE: we could define a generic func here and wrap it in
#     `functools.partial` but it loses its __name__ attribute and
#     having to use `functools.wrap_partial`/@wraps does not totally
#     solve this.


def wght_recall(y_true, y_pred):
    """Similar approach to `class='balanced'`, but ignore dominant class.

    Of the total positives for classes 1 & 2, what is the combined
    recall score?
    """

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    return np.diag(cm)[1:].sum() / cm[1:].sum()


wght_scorer = make_scorer(wght_recall, greater_is_better=True)
