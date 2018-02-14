"""Clean a dataset of Pitchfork album reviews.

WARNING: this script parses around 18,000 HTML trees.
         Runtime is significant but can be made asynchronous.
"""

from concurrent.futures import ProcessPoolExecutor
import os
import re

from bs4 import BeautifulSoup, SoupStrainer
import numpy as np
import pandas as pd
import requests
import unidecode


TEXT_STRAINER = SoupStrainer('div', attrs={'class': 'contents dropcap'})
MULTI_STRAINER = SoupStrainer('nav', attrs={'class': 'album-picker'})
PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

cols = ['reviewid', 'title', 'artist', 'url', 'score', 'best_new_music',
        'author', 'author_type', 'pub_date', 'content', 'release',
        'reissue', 'genres', 'n_genres', 'artists']

dtypes = {
    'reviewid': np.uint16,
    'best_new_music': np.uint8,
    'author': 'category',
    'author_type': 'category',
    'n_genres': np.uint8,
    }

df = pd.read_csv(os.path.join(PATH, 'data/pitchfork.csv'),
                 parse_dates=['pub_date'], index_col='reviewid',
                 infer_datetime_format=True, usecols=cols, dtype=dtypes)

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


# "Best New Music" was launched in 2003.  Drop prior years.
# https://pitchfork.com/best/
df = df[df['pub_date'].dt.year > 2002]
shape_tracker(df, 'Dropped years < 2003')

# We are missing a couple of reviews.  Let's scrape them
null_review_urls = df.loc[df['content'].isnull(), 'url'].tolist()


def extract_review_text(url):
    soup = BeautifulSoup(requests.get(url).text, 'lxml-xml',
                         parse_only=TEXT_STRAINER)
    return ' '.join([p.text for p in soup.find_all('p')]).strip()


df.loc[df['content'].isnull(), 'content'] = [extract_review_text(url)
                                             for url in null_review_urls]

# Drop anything missing that still remains.
# Jet's "Shine On has a 0.0 rating and empty-string review"
# .isnull() doens't pick up on this
df = df[(df['content'] != '') | (df['content'].isnull())]
shape_tracker(df, 'Filtered out null content')

# Force strange Unicode to closest ASCII (smart quotes, em-dashes, etc.)
#    Replace "person's" --> "person" (possession),
#    but convert "that's life" --> "that is life"
#    Convert "don't" --> "dont"
#    This was we can use the default sklearn word tokenizer
regex1 = re.compile("(?P<word>(that|it))'s")
regex2 = re.compile(r"'(?!s)")
regex3 = re.compile(r"'s")
df['content'] = df['content'].apply(
    lambda content: regex3.sub('', regex2.sub('', regex1.sub(
        r'\g<word> is', unidecode.unidecode(content.lower())))))

# We have some "actual" duplicates ie. 9460, 9505
df.drop_duplicates(inplace=True)
shape_tracker(df, 'Dropped duplicates')

# Dummify genre(s).  Currently we have a semicolon-separated strings
#     i.e. pop/r&b; electronic
#     Route below is faster than pd.get_dummies
#     We don't need to drop_first because NaN is implicitly non-null category
# Note: looks like `pandas.Series.str.get_dummies` was recently added...
mask = df['genres'].notnull()
dummies = df.loc[mask, 'genres']\
    .str.split('; ')\
    .apply(set)\
    .apply(lambda x: pd.Series([1] * len(x), index=x))\
    .fillna(0, downcast='infer')\
    .astype(np.uint8)

df = df.merge(right=dummies, how='left', left_index=True, right_index=True)
shape_tracker(df, 'After merge')

# TODO: We have around 8000 unique artists
# Worthwhile to dummify?

# Fill missing genre dummies with 0 (implicit that we dropped some
#     "other" genre)
mask = df['genres'].isnull()
df.loc[mask, 'n_genres':] = df.loc[mask, 'n_genres':].fillna(0.)
df.loc[:, 'n_genres':] = df.loc[:, 'n_genres':].astype(np.uint8)
df.loc[:, 'n_genres'] = df['n_genres'].where(df['n_genres'] != 0, 1)

# We might as well drop `author_type` because it's subject to
#     lookahead bias (only uses current values)
df.drop(['genres', 'author_type'], axis=1, inplace=True)
shape_tracker(df, 'Dropped `genres` and `author_type`')

df.dropna(subset=['title', 'artist', 'release'], inplace=True)
shape_tracker(df, 'Dropped NaN within `title`, `artist`, `release`')

# We have a number of cases of "multi-reviews"
#     https://pitchfork.com/reviews/
#     albums/14817-weezer-pinkerton-deluxe-edition-death-to-false-metal/
#     In this case the 3.5 is being picked up as "best new music"
#     (And, the text of review is for both)
#     We cannot deal with these cases and are forced to just drop them.
#     Use "album-picker" tag class-label
# Also take this opportunity to double-check "best new music" labels
#     and differentiate "Best New Music" v. "Best New Reissue"

MULTI_KEY = b'<nav class="album-picker">'
REISSUE_KEY = b'<p class="bnm-txt">Best new reissue'
ALBUM_KEY = b'<p class="bnm-txt">Best new music'

excep = (TimeoutError,
         ConnectionError,
         requests.exceptions.ConnectionError)


def filter_url(url):
    """Presence or absence of keys in page's content."""
    try:
        resp = requests.get(url, stream=True)
    except excep:
        print('URL timeout:', url)
        return False, False, False
    is_multi = resp.content.find(MULTI_KEY) > -1
    if is_multi:
        # Return early; this record will be dropped from dataset.
        return True, False, False
    is_best_reissue = resp.content.find(REISSUE_KEY) > -1
    is_best_music = resp.content.find(ALBUM_KEY) > -1
    return is_multi, is_best_reissue, is_best_music


def filter_urls(urls):
    """Use pool of processes to send requests asynchronously."""
    with ProcessPoolExecutor() as executor:
        for _, bool_ in zip(urls, executor.map(filter_url, urls)):
            yield bool_


# WARNING: ~10min runtime.  (Reduced from 6 hours...)
#          Can take advantage of multiple cores if OS enables it.
os.system('clear')
print('Scraping reviews.')
is_multi, is_best_reissue, is_best_music = zip(*filter_urls(df['url']))
df.loc[:, 'multi_flag'] = is_multi
df.loc[:, 'best_new_music_src'] = is_best_music
df.loc[:, 'best_new_reissue_src'] = is_best_reissue
df = df.loc[~df['multi_flag']]
shape_tracker(df, 'Dropped mutli-review flags')

assert df['best_new_music'].sum() == \
    df[['best_new_music_src', 'best_new_reissue_src']].values.sum()

print('Writing to %s.' % os.path.join(PATH, 'data/df.pickle'))
df.to_pickle(os.path.join(PATH, 'data/df.pickle'))
