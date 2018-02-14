"""Predict 'Best New Music' label for Pitchfork album reviews."""

# Run on AWS EC2 m5.4xlarge instance

import os
import pickle
from timeit import default_timer
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.classification import UndefinedMetricWarning
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from custom_scorers import wght_recall, wght_scorer  # noqa


RANDOM_STATE = 444
N_JOBS = -1
TEST_SIZE = 0.33
PATH = '/home/brad/pitchfork/'
# PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

df = pd.read_pickle(os.path.join(PATH, 'data/df2.pickle'))
assert df.isnull().values.sum() == 0

y = np.where(df['best_new_reissue_src'], 2,
             np.where(df['best_new_music_src'], 1, 0)).astype(np.uint8)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, train_size=13000, random_state=RANDOM_STATE, stratify=y)
content_train = tuple(X_train['content'])
content_test = tuple(X_test['content'])

# Treat text (`df.content`) and other fields separately.  Implement
#     recursive feature elmination on these fields specifically
#     and mask them out.
# TODO: could probably just throw this into Pipeline
fields = ['n_genres', 'electronic', 'experimental',
          'folk/country', 'global', 'jazz', 'metal', 'pop/r&b', 'rap', 'rock',
          'is_reissue', 'length', 'artist_trend', 'author_trend', 'last_score',
          'delayed', 'spcl_ed', 'unmarked_reissue', 'n_albums',
          'sophomore', 'n_artists']
select = RFE(RandomForestClassifier(n_estimators=250, max_depth=6,
                                    random_state=RANDOM_STATE),
             n_features_to_select=10, step=2)
select.fit(X_train[fields], y_train)
mask = np.asarray(fields)[select.get_support()]
X_train = X_train.loc[:, mask]
X_test = X_test.loc[:, mask]


def grid_search_wrapper(X_train, X_test, y_train, y_test, pipe, param_grid,
                        scoring):
    grid = GridSearchCV(pipe, param_grid, cv=4, n_jobs=N_JOBS,
                        scoring=scoring, return_train_score=False)

    # Bulk of runtime is here
    start = default_timer()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UndefinedMetricWarning)
        grid.fit(X_train, y_train)
    print('Grid search fit in {:0.2f} minutes.'.format(
        (default_timer() - start) / 60))
    return grid


text_kwargs = {'encoding': 'ascii', 'strip_accents': 'ascii',
               'stop_words': 'english', 'max_features': 500}
reg_kwargs = {'max_iter': 500, 'random_state': RANDOM_STATE}
text = [TfidfVectorizer(**text_kwargs)]

# Grid search parameter sequences
c = np.logspace(-5, 3, 9)  # 10**i for i in (-5, 3)
ngrams = ((1, 1), (1, 2))  # (inclusive, inclusive)
min_df = (2, 3)

# These are just throwaway placeholders, really
pipe = {
    'x': Pipeline([('scale', MaxAbsScaler()),
                   ('clf', LogisticRegression(**reg_kwargs))]),
    'content': Pipeline([('text', text[0]),
                         ('scale', MaxAbsScaler()),
                         ('clf', LogisticRegression(**reg_kwargs))])
    }

# Separate parameter grids for each design matrix
params = {
    'x': [
        {'clf': [LogisticRegression(**reg_kwargs)],
         'scale': [MaxAbsScaler()],
         'clf__C': c,
         'clf__class_weight': [None, 'balanced', dict(zip((0, 1, 2),
                                                          (.02, .49, .49)))]},
        {'clf': [RandomForestClassifier(n_estimators=200, n_jobs=N_JOBS)],
         'scale': [None],
         'clf__max_features': ['auto', 'sqrt', 'log2']}
        ],

    'content': [
        {'clf': [LogisticRegression(**reg_kwargs)],
         'clf__C': c,
         'clf__class_weight': [None, 'balanced'],
         'scale': [MaxAbsScaler()],
         'text': text,
         'text__ngram_range': ngrams,
         'text__min_df': min_df},
        {'clf': [MultinomialNB()],
         'scale': [None],
         'text': text,
         'text__ngram_range': ngrams,
         'text__min_df': min_df},
        {'clf': [RandomForestClassifier(n_estimators=500, n_jobs=N_JOBS)],
         'clf__max_features': ['sqrt', 'log2'],
         'scale': [None],
         'text': text,
         'text__ngram_range': ngrams,
         'text__min_df': min_df}
        ]
    }

grids = {
    'x': grid_search_wrapper(X_train, X_test, y_train, y_test, pipe['x'],
                             params['x'], scoring=wght_scorer),
    'content': grid_search_wrapper(content_train, content_test, y_train,
                                   y_test, pipe['content'],
                                   params['content'], scoring=wght_scorer)
    }
# Grid search fit in 0.08 minutes.
# Grid search fit in 11.64 minutes.
# Grid search fit in 0.06 minutes.
# Grid search fit in 11.62 minutes.


data = X_train, X_test, content_train, content_test, y_train, y_test
for obj, path in zip((grids, data), ('grids', 'data')):
    with open(os.path.join(PATH, 'data/%s.pickle' % path), 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

# TODO: SCP these from aws user to local
# scp user@host:~/pitchfork/data/{file}.pickle ~/local/path/{file}.pickle
