/*
Merge table collection from pitchfork.sqlite db and dump to csv
Data source: https://www.kaggle.com/nolanbconaway/pitchfork-data
sqlite> .read merge_pitchfork.sql
*/

.cd /Users/brad/Scripts/python/metis/metisgh/projects/projectmcnulty/pitchfork/
.open pitchfork.sqlite

.mode column
.header on

DROP TABLE IF EXISTS pfork;

CREATE TABLE pfork AS

    SELECT * FROM reviews

    LEFT OUTER JOIN content
    USING (reviewid)

    /* Split original/reissue years; 0=NaN */
    LEFT OUTER JOIN
        (SELECT reviewid, min(year) AS release, (count(*)-1)*max(year) AS reissue FROM years GROUP BY reviewid)
    USING (reviewid)

    /* Concatenate genres with semicolon delimiter */
    LEFT OUTER JOIN
        (SELECT reviewid, group_concat(genre, '; ') AS genres FROM genres GROUP BY reviewid)
    USING (reviewid)

    /* Concatenate artists with semicolon delimiter */
    LEFT OUTER JOIN
        (SELECT reviewid, group_concat(artist, '; ') AS artists from artists GROUP BY reviewid)
    USING (reviewid)

    /* Count number of genres */
    LEFT OUTER JOIN
        (SELECT reviewid, count(genre) as n_genres FROM genres GROUP BY reviewid)
    USING (reviewid);

.mode csv
.once /Users/brad/Scripts/python/metis/metisgh/projects/projectmcnulty/pitchfork/pitchfork.csv

SELECT * FROM pfork;
