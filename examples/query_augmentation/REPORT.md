# Query augmentation: dre under query variations and document noise

Updated 2026-10-13 03:17 — 1396/1397 results.

Query variations after Penha, Câmara & Hauff, *Evaluating the Robustness of Retrieval Pipelines with Query
Variation Generators* (ECIR 2022): one word of every rule / query changed, each generator a separate ablation —
`synonym` (WordNet, most frequent sense), `neighbour_swap`, `random_sub`, `qwerty_sub` (misspellings). RE2RNN: 3
seeds per generator, cells are mean ± std (relative change to the original rules); TREC-5: 1 seed. Methods named
`<method>.<engine>` are dre (matching.py); `phoc_csls100` is the spelling chain PHOC → GloVe 100d scored with
CSLS. Baselines: published RE2RNN numbers (original rules, clean text only), `fuzzy` (`regex` with an edit
budget), `re`, `bm25_word`, `bm25_char4` (TREC-5). Nothing is tuned on these benchmarks.

Caveats: WordNet's most frequent sense is often not the rule's sense (e.g. fare→menu); a replaced word inside
an alternation keeps its siblings, so the rule changes less (see "inside an alternation"); TREC-5 relevance
is the exact occurrence of the original query string, so synonym queries are judged against the original
word's documents and part of their drop is by construction, and many TREC-5 words (names, technical terms)
have no synonym (see "changed").

## RE2RNN zero-shot rule classification — accuracy %

### ATIS

Query variations:

| generator | changed (seed 1) | inside an alternation | examples |
|---|---|---|---|
| synonym | 27/27 | 20/27 | mean→intend, is→exist, planes→aeroplane, airplanes→aeroplane, ticket→tag, flight→flying |
| neighbour_swap | 27/27 | 17/27 | mean→maen, what→waht, planes→plaens, airplanes→ariplanes, and→nad, flight→fligth |
| random_sub | 27/27 | 17/27 | code→codu, what→whaj, airplanes→airpxanes, plane→plaze, costs→cosms, flight→xlight |
| qwerty_sub | 27/27 | 18/27 | mean→meab, what→ahat, planes→planss, number→nkmber, prices→peices, airfare→airfade |

**clean text**

| method | original | synonym | neighbour_swap | random_sub | qwerty_sub |
|---|---|---|---|---|---|
| RE (published) | 87.01 | – | – | – | – |
| FA-RNN (published) | 86.53 | – | – | – | – |
| BiFA-GRU (published) | 88.63 | – | – | – | – |
| exact.viterbi | 85.78 | 65.29 ± 19.40 (-23.9%) | 79.10 ± 2.82 (-7.8%) | 60.54 ± 15.44 (-29.4%) | 60.92 ± 17.93 (-29.0%) |
| levenshtein.viterbi | 87.35 | 84.17 ± 1.55 (-3.6%) | 84.40 ± 0.94 (-3.4%) | 81.97 ± 4.74 (-6.2%) | 82.61 ± 6.48 (-5.4%) |
| phoc.viterbi | 87.12 | 83.58 ± 1.72 (-4.1%) | 86.64 ± 0.59 (-0.6%) | 86.34 ± 0.71 (-0.9%) | 86.94 ± 0.59 (-0.2%) |
| cosine100.viterbi | 35.05 | 33.86 ± 3.51 (-3.4%) | 30.20 ± 3.82 (-13.8%) | 41.25 ± 15.78 (+17.7%) | 25.16 ± 3.56 (-28.2%) |
| csls100.viterbi | 87.23 | 80.14 ± 3.68 (-8.1%) | 85.14 ± 2.11 (-2.4%) | 76.93 ± 6.43 (-11.8%) | 74.62 ± 9.87 (-14.5%) |
| phoc_csls100.viterbi | 87.12 | 80.07 ± 2.70 (-8.1%) | 85.14 ± 1.70 (-2.3%) | 80.03 ± 7.68 (-8.1%) | 86.11 ± 0.40 (-1.2%) |
| exact.dfa | 85.78 | 65.29 ± 19.40 (-23.9%) | 79.10 ± 2.82 (-7.8%) | 60.54 ± 15.44 (-29.4%) | 60.92 ± 17.93 (-29.0%) |
| levenshtein.dfa | 87.57 | 83.91 ± 1.60 (-4.2%) | 84.43 ± 1.05 (-3.6%) | 87.08 ± 0.41 (-0.6%) | 87.42 ± 0.74 (-0.2%) |
| phoc.dfa | 87.01 | 83.69 ± 1.78 (-3.8%) | 86.64 ± 0.74 (-0.4%) | 86.94 ± 0.29 (-0.1%) | 86.94 ± 0.67 (-0.1%) |
| csls100.dfa | 87.23 | 78.91 ± 3.76 (-9.5%) | 84.02 ± 1.68 (-3.7%) | 75.07 ± 6.33 (-13.9%) | 72.71 ± 10.29 (-16.6%) |
| phoc_csls100.dfa | 86.23 | 73.65 ± 9.74 (-14.6%) | 82.34 ± 0.92 (-4.5%) | 83.91 ± 1.78 (-2.7%) | 84.77 ± 0.16 (-1.7%) |

**5% character noise**

| method | original | synonym | neighbour_swap | random_sub | qwerty_sub |
|---|---|---|---|---|---|
| exact.viterbi | 60.81 | 46.73 ± 13.66 (-23.1%) | 56.40 ± 1.45 (-7.2%) | 43.23 ± 10.96 (-28.9%) | 43.64 ± 12.08 (-28.2%) |
| levenshtein.viterbi | 85.78 | 81.60 ± 1.56 (-4.9%) | 83.02 ± 1.04 (-3.2%) | 81.60 ± 3.30 (-4.9%) | 82.23 ± 4.51 (-4.1%) |
| phoc.viterbi | 86.90 | 81.63 ± 1.30 (-6.1%) | 85.82 ± 0.45 (-1.2%) | 85.26 ± 0.59 (-1.9%) | 85.63 ± 1.01 (-1.5%) |
| cosine100.viterbi | 34.60 | 33.41 ± 2.06 (-3.5%) | 31.09 ± 2.02 (-10.1%) | 36.66 ± 8.38 (+5.9%) | 29.04 ± 2.52 (-16.1%) |
| csls100.viterbi | 62.60 | 58.45 ± 2.42 (-6.6%) | 61.48 ± 1.45 (-1.8%) | 55.47 ± 5.11 (-11.4%) | 55.02 ± 6.76 (-12.1%) |
| phoc_csls100.viterbi | 82.53 | 75.66 ± 3.16 (-8.3%) | 80.85 ± 1.62 (-2.0%) | 77.90 ± 5.05 (-5.6%) | 81.93 ± 0.56 (-0.7%) |
| exact.dfa | 60.81 | 46.73 ± 13.66 (-23.1%) | 56.40 ± 1.45 (-7.2%) | 43.23 ± 10.96 (-28.9%) | 43.64 ± 12.08 (-28.2%) |
| levenshtein.dfa | 86.45 | 81.75 ± 1.51 (-5.4%) | 83.50 ± 1.10 (-3.4%) | 85.41 ± 0.28 (-1.2%) | 85.70 ± 1.16 (-0.9%) |
| phoc.dfa | 86.56 | 81.78 ± 1.52 (-5.5%) | 85.70 ± 0.62 (-1.0%) | 85.59 ± 0.32 (-1.1%) | 85.78 ± 1.27 (-0.9%) |
| csls100.dfa | 62.37 | 57.11 ± 2.35 (-8.4%) | 60.10 ± 1.00 (-3.7%) | 54.16 ± 5.07 (-13.2%) | 53.30 ± 7.36 (-14.5%) |
| phoc_csls100.dfa | 81.86 | 70.44 ± 8.14 (-14.0%) | 78.98 ± 0.97 (-3.5%) | 80.18 ± 1.11 (-2.1%) | 80.81 ± 0.42 (-1.3%) |

**20% character noise**

| method | original | synonym | neighbour_swap | random_sub | qwerty_sub |
|---|---|---|---|---|---|
| exact.viterbi | 25.42 | 20.16 ± 5.32 (-20.7%) | 23.29 ± 0.32 (-8.4%) | 18.03 ± 4.03 (-29.1%) | 18.55 ± 4.42 (-27.0%) |
| levenshtein.viterbi | 76.15 | 70.85 ± 3.81 (-7.0%) | 73.87 ± 0.83 (-3.0%) | 73.91 ± 1.54 (-2.9%) | 74.06 ± 2.61 (-2.7%) |
| phoc.viterbi | 78.95 | 73.05 ± 3.58 (-7.5%) | 78.39 ± 0.33 (-0.7%) | 77.01 ± 1.54 (-2.5%) | 76.60 ± 2.10 (-3.0%) |
| cosine100.viterbi | 22.06 | 20.60 ± 0.88 (-6.6%) | 20.12 ± 0.78 (-8.8%) | 20.83 ± 1.27 (-5.6%) | 20.60 ± 1.37 (-6.6%) |
| csls100.viterbi | 26.43 | 24.93 ± 2.02 (-5.6%) | 26.95 ± 0.92 (+2.0%) | 24.00 ± 2.40 (-9.2%) | 24.08 ± 2.79 (-8.9%) |
| phoc_csls100.viterbi | 64.84 | 61.44 ± 2.39 (-5.2%) | 64.91 ± 1.41 (+0.1%) | 65.40 ± 0.18 (+0.9%) | 64.69 ± 1.33 (-0.2%) |
| exact.dfa | 25.42 | 20.16 ± 5.32 (-20.7%) | 23.29 ± 0.32 (-8.4%) | 18.03 ± 4.03 (-29.1%) | 18.55 ± 4.42 (-27.0%) |
| levenshtein.dfa | 77.83 | 71.63 ± 4.02 (-8.0%) | 75.51 ± 0.92 (-3.0%) | 75.33 ± 1.53 (-3.2%) | 75.81 ± 2.16 (-2.6%) |
| phoc.dfa | 78.28 | 72.75 ± 3.51 (-7.1%) | 77.83 ± 0.51 (-0.6%) | 76.41 ± 1.55 (-2.4%) | 76.45 ± 1.97 (-2.3%) |
| csls100.dfa | 26.54 | 24.08 ± 1.02 (-9.3%) | 25.61 ± 0.37 (-3.5%) | 23.40 ± 2.31 (-11.8%) | 23.11 ± 3.21 (-12.9%) |
| phoc_csls100.dfa | 61.93 | 55.92 ± 4.59 (-9.7%) | 60.58 ± 0.90 (-2.2%) | 63.31 ± 1.02 (+2.2%) | 62.60 ± 1.45 (+1.1%) |

### QC (TREC questions)

Query variations:

| generator | changed (seed 1) | inside an alternation | examples |
|---|---|---|---|
| synonym | 60/65 | 36/60 | name→gens, groups→grouping, kind→variety, using→exploitation, called→name, novel→fresh |
| neighbour_swap | 65/65 | 35/65 | name→anme, team→etam, kind→kidn, from→form, called→caleld, novel→nvoel |
| random_sub | 65/65 | 38/65 | name→nxme, group→grozp, kind→kini, using→usrng, called→valled, novel→nxvel |
| qwerty_sub | 65/65 | 37/65 | name→bame, what→whst, what→shat, from→frlm, called→caloed, novel→nofel |

**clean text**

| method | original | synonym | neighbour_swap | random_sub | qwerty_sub |
|---|---|---|---|---|---|
| RE (published) | 64.40 | – | – | – | – |
| FA-RNN (published) | 61.95 | – | – | – | – |
| BiFA-GRU (published) | 62.90 | – | – | – | – |
| exact.viterbi | 60.40 | 54.47 ± 1.05 (-9.8%) | 45.13 ± 2.51 (-25.3%) | 43.27 ± 3.40 (-28.4%) | 44.20 ± 0.71 (-26.8%) |
| levenshtein.viterbi | 62.60 | 59.80 ± 2.12 (-4.5%) | 57.27 ± 0.25 (-8.5%) | 62.67 ± 1.43 (+0.1%) | 63.53 ± 1.98 (+1.5%) |
| phoc.viterbi | 63.20 | 60.27 ± 1.52 (-4.6%) | 61.53 ± 0.74 (-2.6%) | 62.33 ± 1.15 (-1.4%) | 63.93 ± 1.54 (+1.2%) |
| cosine100.viterbi | 46.00 | 37.93 ± 1.33 (-17.5%) | 53.00 ± 4.01 (+15.2%) | 42.00 ± 9.05 (-8.7%) | 47.87 ± 8.77 (+4.1%) |
| csls100.viterbi | 66.40 | 59.87 ± 0.90 (-9.8%) | 53.00 ± 2.57 (-20.2%) | 45.73 ± 4.90 (-31.1%) | 39.33 ± 8.34 (-40.8%) |
| phoc_csls100.viterbi | 67.40 | 63.00 ± 0.65 (-6.5%) | 64.53 ± 0.50 (-4.3%) | 52.87 ± 6.25 (-21.6%) | 48.40 ± 11.20 (-28.2%) |
| exact.dfa | 60.40 | 54.47 ± 1.05 (-9.8%) | 45.13 ± 2.51 (-25.3%) | 43.27 ± 3.40 (-28.4%) | 44.20 ± 0.71 (-26.8%) |
| levenshtein.dfa | 62.60 | 59.67 ± 2.03 (-4.7%) | 56.87 ± 0.84 (-9.2%) | 62.60 ± 1.45 (+0.0%) | 63.60 ± 2.16 (+1.6%) |
| phoc.dfa | 63.40 | 59.67 ± 1.68 (-5.9%) | 62.00 ± 0.91 (-2.2%) | 63.00 ± 0.91 (-0.6%) | 63.93 ± 1.27 (+0.8%) |
| csls100.dfa | 71.00 | 66.00 ± 0.71 (-7.0%) | 55.47 ± 4.08 (-21.9%) | 45.93 ± 5.72 (-35.3%) | 41.33 ± 8.67 (-41.8%) |
| phoc_csls100.dfa | 50.80 | 57.93 ± 9.29 (+14.0%) | 52.33 ± 8.89 (+3.0%) | 52.87 ± 5.47 (+4.1%) | 47.87 ± 11.02 (-5.8%) |

**5% character noise**

| method | original | synonym | neighbour_swap | random_sub | qwerty_sub |
|---|---|---|---|---|---|
| exact.viterbi | 53.00 | 47.20 ± 0.86 (-10.9%) | 39.27 ± 1.61 (-25.9%) | 38.47 ± 1.98 (-27.4%) | 38.53 ± 0.66 (-27.3%) |
| levenshtein.viterbi | 61.20 | 58.40 ± 2.04 (-4.6%) | 56.00 ± 1.14 (-8.5%) | 61.13 ± 0.98 (-0.1%) | 61.20 ± 1.23 (+0.0%) |
| phoc.viterbi | 61.40 | 59.47 ± 1.73 (-3.1%) | 60.00 ± 0.75 (-2.3%) | 61.13 ± 0.81 (-0.4%) | 62.07 ± 0.66 (+1.1%) |
| cosine100.viterbi | 41.80 | 33.53 ± 1.43 (-19.8%) | 43.87 ± 3.09 (+4.9%) | 33.20 ± 7.25 (-20.6%) | 40.93 ± 7.06 (-2.1%) |
| csls100.viterbi | 55.20 | 48.80 ± 1.18 (-11.6%) | 44.13 ± 3.03 (-20.0%) | 38.40 ± 4.17 (-30.4%) | 34.07 ± 5.23 (-38.3%) |
| phoc_csls100.viterbi | 60.00 | 55.53 ± 0.34 (-7.4%) | 55.67 ± 0.66 (-7.2%) | 46.07 ± 4.63 (-23.2%) | 42.20 ± 8.67 (-29.7%) |
| exact.dfa | 53.00 | 47.20 ± 0.86 (-10.9%) | 39.27 ± 1.61 (-25.9%) | 38.47 ± 1.98 (-27.4%) | 38.53 ± 0.66 (-27.3%) |
| levenshtein.dfa | 61.00 | 58.53 ± 2.02 (-4.0%) | 55.67 ± 1.32 (-8.7%) | 60.87 ± 1.23 (-0.2%) | 61.27 ± 1.23 (+0.4%) |
| phoc.dfa | 61.80 | 58.73 ± 1.72 (-5.0%) | 60.13 ± 1.00 (-2.7%) | 61.67 ± 0.62 (-0.2%) | 62.00 ± 0.59 (+0.3%) |
| csls100.dfa | 57.60 | 53.00 ± 0.00 (-8.0%) | 46.53 ± 3.31 (-19.2%) | 38.87 ± 3.92 (-32.5%) | 36.40 ± 5.77 (-36.8%) |
| phoc_csls100.dfa | 47.80 | 52.20 ± 6.38 (+9.2%) | 46.27 ± 6.18 (-3.2%) | 46.13 ± 4.43 (-3.5%) | 41.80 ± 9.04 (-12.6%) |

**20% character noise**

| method | original | synonym | neighbour_swap | random_sub | qwerty_sub |
|---|---|---|---|---|---|
| exact.viterbi | 38.80 | 36.73 ± 0.52 (-5.3%) | 31.40 ± 0.71 (-19.1%) | 30.73 ± 0.50 (-20.8%) | 30.87 ± 0.25 (-20.4%) |
| levenshtein.viterbi | 58.20 | 52.67 ± 1.47 (-9.5%) | 52.53 ± 1.64 (-9.7%) | 56.53 ± 0.38 (-2.9%) | 56.33 ± 0.09 (-3.2%) |
| phoc.viterbi | 57.60 | 53.20 ± 1.42 (-7.6%) | 55.93 ± 0.34 (-2.9%) | 56.00 ± 0.43 (-2.8%) | 56.07 ± 0.41 (-2.7%) |
| cosine100.viterbi | 26.60 | 21.87 ± 1.68 (-17.8%) | 25.93 ± 0.19 (-2.5%) | 19.73 ± 2.81 (-25.8%) | 24.60 ± 4.95 (-7.5%) |
| csls100.viterbi | 35.60 | 32.93 ± 0.66 (-7.5%) | 31.13 ± 4.20 (-12.5%) | 25.80 ± 3.12 (-27.5%) | 23.00 ± 3.37 (-35.4%) |
| phoc_csls100.viterbi | 40.00 | 37.47 ± 0.81 (-6.3%) | 38.53 ± 1.59 (-3.7%) | 35.00 ± 3.96 (-12.5%) | 31.07 ± 5.19 (-22.3%) |
| exact.dfa | 38.80 | 36.73 ± 0.52 (-5.3%) | 31.40 ± 0.71 (-19.1%) | 30.73 ± 0.50 (-20.8%) | 30.87 ± 0.25 (-20.4%) |
| levenshtein.dfa | 57.80 | 52.60 ± 1.23 (-9.0%) | 51.60 ± 2.27 (-10.7%) | 56.40 ± 0.71 (-2.4%) | 56.27 ± 0.34 (-2.7%) |
| phoc.dfa | 57.60 | 52.73 ± 1.27 (-8.4%) | 55.27 ± 0.94 (-4.1%) | 55.80 ± 0.43 (-3.1%) | 55.80 ± 0.99 (-3.1%) |
| csls100.dfa | 39.20 | 35.73 ± 0.62 (-8.8%) | 32.07 ± 4.21 (-18.2%) | 26.40 ± 3.15 (-32.7%) | 23.73 ± 3.17 (-39.5%) |
| phoc_csls100.dfa | 38.60 | 37.40 ± 2.55 (-3.1%) | 34.33 ± 2.38 (-11.1%) | 35.07 ± 2.96 (-9.2%) | 32.00 ± 5.38 (-17.1%) |

### SMS spam

Query variations:

| generator | changed (seed 1) | inside an alternation | examples |
|---|---|---|---|
| synonym | 33/35 | 11/33 | call→vociferation, call→yell, prize→award, winner→victor, free→liberate, call→outcry |
| neighbour_swap | 34/35 | 9/34 | call→acll, ringtone→rintgone, won→wno, winner→iwnner, free→rfee, dating→daitng |
| random_sub | 34/35 | 11/34 | call→rall, ringtone→rinptone, prize→pfize, reward→rpward, guaranteed→guaranfeed, dating→dajing |
| qwerty_sub | 34/35 | 5/34 | call→cxll, call→calo, cash→caqh, reward→rewars, guaranteed→guaranreed, call→calk |

**clean text**

| method | original | synonym | neighbour_swap | random_sub | qwerty_sub |
|---|---|---|---|---|---|
| RE (published) | 93.20 | – | – | – | – |
| FA-RNN (published) | 93.00 | – | – | – | – |
| BiFA-GRU (published) | 93.20 | – | – | – | – |
| exact.viterbi | 93.20 | 86.20 ± 0.28 (-7.5%) | 86.67 ± 0.09 (-7.0%) | 86.67 ± 0.09 (-7.0%) | 86.93 ± 0.09 (-6.7%) |
| levenshtein.viterbi | 93.20 | 86.00 ± 0.43 (-7.7%) | 86.60 ± 0.16 (-7.1%) | 93.33 ± 0.09 (+0.1%) | 93.53 ± 0.19 (+0.4%) |
| phoc.viterbi | 93.20 | 86.13 ± 0.34 (-7.6%) | 93.33 ± 0.09 (+0.1%) | 93.20 ± 0.16 (+0.0%) | 93.53 ± 0.19 (+0.4%) |
| cosine100.viterbi | 19.80 | 19.07 ± 1.23 (-3.7%) | 64.27 ± 5.36 (+224.6%) | 78.67 ± 9.13 (+297.3%) | 80.27 ± 7.35 (+305.4%) |
| csls100.viterbi | 93.40 | 86.47 ± 0.09 (-7.4%) | 86.73 ± 0.19 (-7.1%) | 86.67 ± 0.09 (-7.2%) | 87.00 ± 0.16 (-6.9%) |
| phoc_csls100.viterbi | 94.00 | 85.73 ± 0.34 (-8.8%) | 94.27 ± 0.19 (+0.3%) | 89.93 ± 1.27 (-4.3%) | 90.20 ± 1.85 (-4.0%) |
| exact.dfa | 93.20 | 86.20 ± 0.28 (-7.5%) | 86.67 ± 0.09 (-7.0%) | 86.67 ± 0.09 (-7.0%) | 86.93 ± 0.09 (-6.7%) |
| levenshtein.dfa | 93.20 | 86.00 ± 0.43 (-7.7%) | 86.60 ± 0.16 (-7.1%) | 93.33 ± 0.09 (+0.1%) | 93.53 ± 0.19 (+0.4%) |
| phoc.dfa | 92.60 | 86.00 ± 0.43 (-7.1%) | 93.33 ± 0.09 (+0.8%) | 93.00 ± 0.16 (+0.4%) | 93.40 ± 0.16 (+0.9%) |
| csls100.dfa | 93.40 | 84.13 ± 1.52 (-9.9%) | 86.87 ± 0.09 (-7.0%) | 86.67 ± 0.09 (-7.2%) | 87.13 ± 0.34 (-6.7%) |
| phoc_csls100.dfa | 83.20 | 67.00 ± 2.98 (-19.5%) | 78.00 ± 0.59 (-6.3%) | 75.40 ± 1.28 (-9.4%) | 80.27 ± 5.96 (-3.5%) |

**5% character noise**

| method | original | synonym | neighbour_swap | random_sub | qwerty_sub |
|---|---|---|---|---|---|
| exact.viterbi | 91.40 | 86.33 ± 0.19 (-5.5%) | 86.67 ± 0.09 (-5.2%) | 86.73 ± 0.19 (-5.1%) | 86.93 ± 0.09 (-4.9%) |
| levenshtein.viterbi | 93.00 | 86.20 ± 0.43 (-7.3%) | 86.53 ± 0.09 (-7.0%) | 92.13 ± 0.25 (-0.9%) | 91.93 ± 0.34 (-1.1%) |
| phoc.viterbi | 92.60 | 86.00 ± 0.43 (-7.1%) | 91.80 ± 0.16 (-0.9%) | 92.33 ± 0.38 (-0.3%) | 92.33 ± 0.34 (-0.3%) |
| cosine100.viterbi | 21.80 | 20.60 ± 1.41 (-5.5%) | 67.60 ± 6.67 (+210.1%) | 81.93 ± 6.90 (+275.8%) | 82.47 ± 4.78 (+278.3%) |
| csls100.viterbi | 91.40 | 86.60 ± 0.00 (-5.3%) | 86.93 ± 0.25 (-4.9%) | 86.73 ± 0.19 (-5.1%) | 86.93 ± 0.09 (-4.9%) |
| phoc_csls100.viterbi | 92.80 | 86.13 ± 0.62 (-7.2%) | 93.60 ± 0.33 (+0.9%) | 89.40 ± 0.82 (-3.7%) | 90.13 ± 1.18 (-2.9%) |
| exact.dfa | 91.40 | 86.33 ± 0.19 (-5.5%) | 86.67 ± 0.09 (-5.2%) | 86.73 ± 0.19 (-5.1%) | 86.93 ± 0.09 (-4.9%) |
| levenshtein.dfa | 93.00 | 86.20 ± 0.43 (-7.3%) | 86.53 ± 0.09 (-7.0%) | 92.13 ± 0.25 (-0.9%) | 91.93 ± 0.34 (-1.1%) |
| phoc.dfa | 92.20 | 85.87 ± 0.52 (-6.9%) | 91.80 ± 0.16 (-0.4%) | 92.07 ± 0.47 (-0.1%) | 92.27 ± 0.25 (+0.1%) |
| csls100.dfa | 91.80 | 84.80 ± 1.14 (-7.6%) | 87.13 ± 0.25 (-5.1%) | 86.73 ± 0.19 (-5.5%) | 87.00 ± 0.16 (-5.2%) |
| phoc_csls100.dfa | 84.20 | 68.13 ± 2.89 (-19.1%) | 79.53 ± 0.57 (-5.5%) | 78.20 ± 1.13 (-7.1%) | 80.40 ± 5.77 (-4.5%) |

**20% character noise**

| method | original | synonym | neighbour_swap | random_sub | qwerty_sub |
|---|---|---|---|---|---|
| exact.viterbi | 89.20 | 86.47 ± 0.09 (-3.1%) | 86.60 ± 0.00 (-2.9%) | 86.67 ± 0.09 (-2.8%) | 86.67 ± 0.19 (-2.8%) |
| levenshtein.viterbi | 92.20 | 86.33 ± 0.47 (-6.4%) | 86.67 ± 0.09 (-6.0%) | 89.73 ± 0.09 (-2.7%) | 90.00 ± 0.16 (-2.4%) |
| phoc.viterbi | 92.80 | 86.20 ± 0.57 (-7.1%) | 90.00 ± 0.28 (-3.0%) | 90.40 ± 0.33 (-2.6%) | 90.47 ± 0.25 (-2.5%) |
| cosine100.viterbi | 31.20 | 26.07 ± 3.21 (-16.5%) | 75.87 ± 3.64 (+143.2%) | 82.87 ± 3.19 (+165.6%) | 84.07 ± 2.58 (+169.4%) |
| csls100.viterbi | 89.20 | 86.33 ± 0.25 (-3.2%) | 86.67 ± 0.09 (-2.8%) | 86.67 ± 0.09 (-2.8%) | 86.67 ± 0.19 (-2.8%) |
| phoc_csls100.viterbi | 90.60 | 85.07 ± 0.41 (-6.1%) | 90.47 ± 0.25 (-0.1%) | 88.13 ± 0.57 (-2.7%) | 88.33 ± 0.77 (-2.5%) |
| exact.dfa | 89.20 | 86.47 ± 0.09 (-3.1%) | 86.60 ± 0.00 (-2.9%) | 86.67 ± 0.09 (-2.8%) | 86.67 ± 0.19 (-2.8%) |
| levenshtein.dfa | 92.20 | 86.33 ± 0.47 (-6.4%) | 86.67 ± 0.09 (-6.0%) | 89.73 ± 0.09 (-2.7%) | 90.00 ± 0.16 (-2.4%) |
| phoc.dfa | 92.60 | 86.00 ± 0.57 (-7.1%) | 90.07 ± 0.25 (-2.7%) | 90.53 ± 0.34 (-2.2%) | 90.47 ± 0.34 (-2.3%) |
| csls100.dfa | 89.00 | 85.07 ± 1.00 (-4.4%) | 86.67 ± 0.09 (-2.6%) | 86.67 ± 0.09 (-2.6%) | 86.67 ± 0.19 (-2.6%) |
| phoc_csls100.dfa | 86.80 | 72.80 ± 2.70 (-16.1%) | 83.93 ± 0.25 (-3.3%) | 80.20 ± 1.80 (-7.6%) | 82.67 ± 4.19 (-4.8%) |

## TREC-5 Confusion track — MAP

### TREC-5 Confusion track

Query variations:

| generator | changed (seed 1) | inside an alternation | examples |
|---|---|---|---|
| synonym | 87/120 | 4/87 | laggard→dawdler, seawall→mole, cheetah→chetah, plumage→feather, typified→epitomise, spinners→spinster |
| neighbour_swap | 120/120 | 5/120 | laggard→laggrad, isopropoxyphenyl→isopropoxypehnyl, sirenia→sireina, sidestream→sidestrema, seawall→saewall, submarket→sumbarket |
| random_sub | 120/120 | 6/120 | laggard→waggard, isopropoxyphenyl→isoprxpoxyphenyl, sirenia→sirbnia, sidestream→shdestream, seawall→stawall, submarket→submjrket |
| qwerty_sub | 120/120 | 5/120 | laggard→kaggard, isopropoxyphenyl→isopropoxypmenyl, sirenia→sirenix, sidestream→siddstream, seawall→seawqll, submarket→submarkey |

**original documents**

| method | original | synonym | neighbour_swap | random_sub | qwerty_sub |
|---|---|---|---|---|---|
| bm25_word | 0.7782 | 0.3814 (-51.0%) | 0.1258 (-83.8%) | 0.1268 (-83.7%) | 0.1186 (-84.8%) |
| bm25_char4 | 0.5436 | 0.3216 (-40.8%) | 0.2563 (-52.8%) | 0.3451 (-36.5%) | 0.3174 (-41.6%) |
| re | 1.0000 | 0.2929 (-70.7%) | 0.0083 (-99.2%) | 0.0528 (-94.7%) | 0.0315 (-96.8%) |
| fuzzy | 1.0000 | 0.3877 (-61.2%) | 0.5169 (-48.3%) | 0.9407 (-5.9%) | 0.9153 (-8.5%) |
| exact.viterbi | 1.0000 | 0.2925 (-70.7%) | 0.0083 (-99.2%) | 0.0528 (-94.7%) | 0.0315 (-96.8%) |
| levenshtein.viterbi | 0.9997 | 0.4465 (-55.3%) | 0.7809 (-21.9%) | 0.9494 (-5.0%) | 0.9220 (-7.8%) |
| phoc.viterbi | 0.9989 | 0.4069 (-59.3%) | 0.9813 (-1.8%) | 0.9536 (-4.5%) | 0.9248 (-7.4%) |
| phoc_csls100.viterbi | 0.9997 | 0.4982 (-50.2%) | 0.7195 (-28.0%) | 0.7556 (-24.4%) | 0.7309 (-26.9%) |

**OCR, 5% character errors**

| method | original | synonym | neighbour_swap | random_sub | qwerty_sub |
|---|---|---|---|---|---|
| bm25_word | 0.5479 | 0.2329 (-57.5%) | 0.0976 (-82.2%) | 0.0892 (-83.7%) | 0.0897 (-83.6%) |
| bm25_char4 | 0.5123 | 0.2736 (-46.6%) | 0.2156 (-57.9%) | 0.3126 (-39.0%) | 0.3090 (-39.7%) |
| re | 0.6331 | 0.1747 (-72.4%) | 0.0000 (-100.0%) | 0.0111 (-98.2%) | 0.0232 (-96.3%) |
| exact.viterbi | 0.6328 | 0.1744 (-72.4%) | 0.0000 (-100.0%) | 0.0111 (-98.2%) | 0.0232 (-96.3%) |
| levenshtein.viterbi | 0.9594 | 0.4232 (-55.9%) | 0.6873 (-28.4%) | 0.8952 (-6.7%) | 0.8692 (-9.4%) |
| phoc.viterbi | 0.9482 | 0.3846 (-59.4%) | 0.9112 (-3.9%) | 0.8806 (-7.1%) | 0.8531 (-10.0%) |
| phoc_csls100.viterbi | 0.8614 | 0.4170 (-51.6%) | 0.6208 (-27.9%) | 0.6595 (-23.4%) | 0.6525 (-24.2%) |

**OCR, 20% character errors**

| method | original | synonym | neighbour_swap | random_sub | qwerty_sub |
|---|---|---|---|---|---|
| bm25_word | 0.2383 | 0.1105 (-53.6%) | 0.0433 (-81.8%) | 0.0575 (-75.9%) | 0.0540 (-77.3%) |
| bm25_char4 | 0.2771 | 0.1526 (-44.9%) | 0.1221 (-55.9%) | 0.1706 (-38.4%) | 0.1577 (-43.1%) |
| re | 0.1674 | 0.0844 (-49.6%) | 0.0001 (-100.0%) | 0.0015 (-99.1%) | 0.0002 (-99.9%) |
| exact.viterbi | 0.1645 | 0.0824 (-49.9%) | 0.0001 (-100.0%) | 0.0015 (-99.1%) | 0.0002 (-99.9%) |
| levenshtein.viterbi | 0.4643 | 0.2143 (-53.8%) | 0.2810 (-39.5%) | 0.3943 (-15.1%) | 0.3789 (-18.4%) |
| phoc.viterbi | 0.4259 | 0.1851 (-56.5%) | 0.3845 (-9.7%) | 0.3597 (-15.6%) | 0.3633 (-14.7%) |
| phoc_csls100.viterbi | 0.3338 | … | 0.2444 (-26.8%) | 0.2435 (-27.1%) | 0.2337 (-30.0%) |

