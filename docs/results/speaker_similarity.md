Speakers/recordings: 5  (small sample: indicative only)

| comparison | cosine similarity | trials above threshold |
|---|---|---|
| same speaker, original vs original | 0.791 (min 0.734, max 0.953, n=5) | 5/5 |
| different speakers, original | 0.498 (min 0.346, max 0.620, n=20) | 0/20 |
| natural: original vs processed (same speaker) | 0.691 (min 0.568, max 0.849, n=5) | 3/5 |
| natural: processed vs processed (same speaker) | 0.753 (min 0.640, max 0.938, n=5) | 4/5 |
| natural: processed, different speakers | 0.528 (min 0.420, max 0.684, n=20) | 3/20 |
| balanced: original vs processed (same speaker) | 0.642 (min 0.506, max 0.789, n=5) | 2/5 |
| balanced: processed vs processed (same speaker) | 0.751 (min 0.636, max 0.951, n=5) | 4/5 |
| balanced: processed, different speakers | 0.531 (min 0.424, max 0.687, n=20) | 3/20 |
| strong: original vs processed (same speaker) | 0.604 (min 0.498, max 0.711, n=5) | 2/5 |
| strong: processed vs processed (same speaker) | 0.780 (min 0.688, max 0.950, n=5) | 5/5 |
| strong: processed, different speakers | 0.549 (min 0.406, max 0.735, n=20) | 4/20 |

Threshold (midpoint of original same/different means): 0.645

Per recording, original-vs-processed cosine:
| recording | same-speaker baseline | natural | balanced | strong |
|---|---|---|---|---|
| arctic_a0007_male | 0.734 | 0.568 | 0.506 | 0.498 |
| arctic_aew_male | 0.777 | 0.747 | 0.713 | 0.693 |
| arctic_axb_female | 0.735 | 0.660 | 0.633 | 0.578 |
| mssnsd_clnsp1_male | 0.953 | 0.849 | 0.789 | 0.711 |
| speechbrain_example1 | 0.758 | 0.630 | 0.569 | 0.537 |
