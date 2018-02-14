"""Feature engineering routine on dataset of Pitchfork album reviews."""

import os

import pandas as pd
import numpy as np


PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

df = pd.read_pickle(os.path.join(PATH, 'data/df.pickle'))
print('Raw shape:', df.shape)
print()
shape = [df.shape]


def shape_tracker(df, msg=None):
    """Helper to report evolving shape of `df` to STDOUT."""
    global shape
    newshape = df.shape
    if not shape:
        return
    oldshape = shape[-1]
    rows, cols = [oldshape[i] - newshape[i] for i in [0, 1]]
    if msg:
        print('{msg}:'.format(msg=msg))
    print('Dropped {} row(s) & {} column(s).'.format(rows, cols))
    print('New shape:', df.shape)
    print()
    shape.append(newshape)


# How can we bin `score`?  3 alternations here.
# Ultimately, may try to exclude score completely from design matrix.
df.loc[:, 'score_quartile'] = pd.qcut(df['score'], 4, labels=range(1, 5))
df.loc[:, 'score_quintile'] = pd.qcut(df['score'], 5, labels=range(1, 6))

threshold = 9.0
df.loc[:, 'score_binary'] = np.where(df['score'].ge(threshold), 1, 0)\
    .astype(np.uint8)
shape_tracker(df, 'Binned `score`')

# Derive a few simple fields
df.loc[:, 'is_reissue'] = df['reissue'].where(df['reissue'] == 0., 1.)\
    .astype(np.uint8)
df.loc[:, 'length'] = df['content'].str.len()
df.loc[:, 'n_artists'] = df['artists'].str.count('; ').add(1).astype(np.uint8)
shape_tracker(df, 'Derived fields')

# Time-series derivations
# ---------------------------------------------------------------------


def stack_strings(ser, pat='; '):
    stacked = ser.str.split(pat=pat, expand=True).stack()
    stacked.index = stacked.index.droplevel(1)
    return stacked.to_frame()  # leave name blank


def score_trend(df, ser, sortby: str, var: str, pat='; ', fillvalue=0.,
                name=None):
    _name = ser.name
    # "Buffer dtype mismatch, expected 'Python object' but got 'long'""
    #     This is caused by the merge with pd.Categorical dtypes
    #     and is okay in this case.
    # https://github.com/pandas-dev/pandas/issues/18646
    # Should be fixed in v 0.23.0
    stacked = stack_strings(ser, pat=pat)\
        .merge(df, left_index=True, right_index=True)\
        .drop(_name, axis=1)\
        .rename(columns={0: _name})\
        .sort_values(sortby)\
        .groupby(_name, sort=False).expanding()[var].mean()\
        .groupby(level=_name, sort=False).shift(1).fillna(fillvalue)
    return stacked.groupby(stacked.index.get_level_values(1),
                           sort=False).mean().to_frame(name=name)


def rank_last(df, ser, sortby: str, var: str, name: str, pat='; ',
              fillvalue=0.):
    _name = ser.name
    stacked = stack_strings(ser, pat=pat)\
        .merge(df, left_index=True, right_index=True)\
        .drop(_name, axis=1)\
        .rename(columns={0: _name})\
        .sort_values([_name, sortby])
    mask = stacked[_name].ne(stacked[_name].shift())
    stacked[name] = np.where(mask, np.nan, stacked[var].shift())
    return stacked.groupby(level=0)[name].mean().fillna(fillvalue).to_frame()


# Cumulative mean of albums where artist was present
#     up to but not excl. current.
#     If not avail, fill with arbitrary fill value.
df.sort_values('pub_date', inplace=True)
fillvalue = 4.0
artist_trend = score_trend(df, df['artists'], 'pub_date', 'score',
                           fillvalue=fillvalue, name='artist_trend')

# Same as above, but for review author
writer_trend = score_trend(df, df['author'], 'pub_date', 'score',
                           fillvalue=fillvalue, name='author_trend')

# A shorter-term "hype factor" -
#     value of last album where this artist was present.  (By pub date.)
#     Default to the same `fillvalue`.
last_score = rank_last(df, ser=df['artists'], sortby='pub_date', var='score',
                       name='last_score', fillvalue=fillvalue)

params = dict(left_index=True, right_index=True)
df = df.merge(artist_trend, **params)\
       .merge(writer_trend, **params)\
       .merge(last_score, **params)

# Undo these scores for "various artists" cases
mask = df['artists'] == 'various artists'
df.loc[mask, ['artist_trend', 'last_score']] = fillvalue

# Cumulative count of albums (single-artist cases only)
df['_temp'] = 1.  # placeholder for cumulative count
mask = (~df.artists.str.contains('; ')) & (df.artist != 'various artists')
df.loc[mask, 'n_albums'] = df.loc[mask].sort_values(['release', 'pub_date'])\
    .groupby('artists')['_temp']\
    .cumsum()
df.loc[:, 'n_albums'] = df['n_albums'].fillna(1).astype(np.uint8)
del df['_temp']

# Sophomore slump?
df.loc[:, 'sophomore'] = np.where(df['n_albums'] == 2, 1, 0).astype(np.uint8)

# Old albums that are *not* technically reissues but are reviewed "late"
#     have 0 chance of being Best New Music, but are often highly praised.
#     http://pitchfork.com/reviews/albums/21845-sign-o-the-times/
# These aren't marked as reissues either.
df.loc[:, 'span'] = df['pub_date'].dt.year - df['release']
df.loc[:, 'delayed'] = \
    ((df['span'].gt(1)) & (df['is_reissue'].ne(1))).astype(np.uint8)
df.loc[:, 'spcl_ed'] = df['title'].str.contains(
    r'(anniversary|deluxe|legacy)(?: edition)?', case=False).astype(np.uint8)

# Reissue that is not explicitly marked as one
df.loc[:, 'unmarked_reissue'] = \
    ((df['spcl_ed'].eq(1)) & (df['is_reissue'].eq(0))).astype(np.uint8)

print('Writing to %s.' % os.path.join(PATH, 'data/df2.pickle'))
df.to_pickle(os.path.join(PATH, 'data/df2.pickle'))
