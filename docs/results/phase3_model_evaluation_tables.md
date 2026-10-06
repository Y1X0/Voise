# Phase 3 raw tables (generated)

Main run: scripts/neural_anonymization_eval.py (DSP, psn_world, kNN-VC). B3 scored in a separate run with the same split/seeds/evaluators.

TEST: 9 unseen speakers, 27 utterances. Split: {'train': ['ami-FEE078', 'ami-FEE087', 'arctic-aew-male', 'mssnsd-clnsp1-male'], 'val': ['ami-MEE009', 'pyannote-sample-speaker91'], 'test': ['ami-FEE083', 'ami-FEO070', 'ami-MEE012', 'ami-MEE075', 'ami-MÉO069', 'arctic-a0007-male', 'arctic-axb-female', 'pyannote-sample-speaker90', 'speechbrain-example1']}

### Evaluator: ge2e  (original->original EER 5.1 % [0-9], top-1 93 %)

| system | ignorant EER | lazy same-session EER | lazy cross-session EER A→B / A→C / B→C | lazy cross + WCCN | cross A→B top-1 | same proc↔proc (A→B) mean / max | diff proc↔proc mean |
|---|---|---|---|---|---|---|---|
| dsp_natural | 7.7 % [4-17] | 7.7 % [0-13] | 6.3 % [0-11] / 6.4 % [1-12] / 6.1 % [0-11] | 6.4 % [0-12] | 93 % | 0.820 / 0.942 | 0.588 |
| dsp_balanced | 11.6 % [7-23] | 5.1 % [0-11] | 4.2 % [0-11] / 5.1 % [1-11] / 5.1 % [1-12] | 5.0 % [0-11] | 93 % | 0.821 / 0.919 | 0.585 |
| dsp_strong | 17.9 % [11-31] | 4.5 % [0-14] | 3.8 % [0-12] / 3.8 % [1-13] / 5.1 % [1-12] | 4.2 % [0-12] | 93 % | 0.818 / 0.922 | 0.586 |
| psn_world | 11.5 % [6-20] | 10.3 % [0-19] | 14.1 % [4-22] / 14.4 % [5-23] / 11.5 % [4-20] | 13.1 % [4-22] | 81 % | 0.772 / 0.892 | 0.596 |
| psn_world_cmvn | 11.8 % [7-29] | 13.1 % [1-21] | 14.2 % [4-20] / 15.4 % [5-21] / 12.8 % [3-21] | 14.4 % [4-20] | 85 % | 0.774 / 0.992 | 0.578 |
| knnvc_pseudo | 29.5 % [19-39] | 33.3 % [18-41] | 33.6 % [24-43] / 35.9 % [23-45] / 30.8 % [18-39] | 33.4 % [23-45] | 30 % | 0.783 / 0.874 | 0.733 |

### Evaluator: mfcc_stats  (original->original EER 20.0 % [3-28], top-1 74 %)

| system | ignorant EER | lazy same-session EER | lazy cross-session EER A→B / A→C / B→C | lazy cross + WCCN | cross A→B top-1 | same proc↔proc (A→B) mean / max | diff proc↔proc mean |
|---|---|---|---|---|---|---|---|
| dsp_natural | 28.4 % [18-35] | 28.2 % [13-40] | 28.2 % [14-39] / 28.1 % [14-37] / 28.2 % [15-36] | 26.7 % [16-33] | 63 % | 0.388 / 0.713 | 0.168 |
| dsp_balanced | 34.7 % [24-44] | 25.6 % [14-37] | 26.6 % [14-37] / 25.3 % [14-38] / 25.6 % [15-37] | 24.4 % [14-33] | 74 % | 0.399 / 0.773 | 0.174 |
| dsp_strong | 40.7 % [30-56] | 28.2 % [11-39] | 27.0 % [10-39] / 25.6 % [11-37] / 25.9 % [11-38] | 28.2 % [11-35] | 67 % | 0.406 / 0.771 | 0.180 |
| psn_world | 24.4 % [19-37] | 20.7 % [11-31] | 20.5 % [11-32] / 20.8 % [13-32] / 20.7 % [11-35] | 19.2 % [11-30] | 63 % | 0.381 / 0.668 | 0.098 |
| psn_world_cmvn | 34.6 % [27-46] | 28.5 % [10-37] | 28.4 % [12-38] / 29.6 % [14-39] / 28.4 % [8-38] | 28.2 % [14-38] | 63 % | 0.313 / 0.847 | 0.091 |
| knnvc_pseudo | 47.4 % [42-54] | 41.0 % [24-49] | 42.5 % [31-49] / 44.9 % [32-53] / 43.6 % [33-54] | 40.9 % [28-50] | 15 % | 0.542 / 0.741 | 0.508 |

### Evaluator: vpc_ecapa  (original->original EER 0.6 % [0-4], top-1 100 %)

| system | ignorant EER | lazy same-session EER | lazy cross-session EER A→B / A→C / B→C | lazy cross + WCCN | cross A→B top-1 | same proc↔proc (A→B) mean / max | diff proc↔proc mean |
|---|---|---|---|---|---|---|---|
| dsp_natural | 7.7 % [1-13] | 2.6 % [0-7] | 3.6 % [0-6] / 2.3 % [0-5] / 2.5 % [0-5] | 2.9 % [0-5] | 100 % | 0.659 / 0.842 | 0.196 |
| dsp_balanced | 11.5 % [6-20] | 2.6 % [0-5] | 2.6 % [0-5] / 2.6 % [0-7] / 2.9 % [0-7] | 2.7 % [0-5] | 100 % | 0.659 / 0.815 | 0.189 |
| dsp_strong | 19.2 % [11-28] | 2.6 % [0-5] | 1.3 % [0-3] / 1.4 % [0-4] / 2.5 % [0-4] | 2.5 % [0-4] | 100 % | 0.633 / 0.828 | 0.191 |
| psn_world | 5.2 % [0-13] | 2.6 % [0-7] | 2.9 % [0-8] / 3.8 % [0-9] / 2.6 % [0-11] | 2.9 % [0-8] | 100 % | 0.662 / 0.792 | 0.234 |
| psn_world_cmvn | 6.4 % [2-15] | 5.1 % [0-10] | 7.5 % [0-13] / 6.4 % [0-13] / 9.1 % [0-15] | 5.1 % [0-11] | 100 % | 0.648 / 0.834 | 0.254 |
| knnvc_pseudo | 32.4 % [24-44] | 25.6 % [16-37] | 30.9 % [18-41] / 30.9 % [18-45] / 28.3 % [16-38] | 28.5 % [17-40] | 26 % | 0.621 / 0.784 | 0.530 |

### Evaluator: resnet_vox1  (original->original EER 0.3 % [0-5], top-1 96 %)

| system | ignorant EER | lazy same-session EER | lazy cross-session EER A→B / A→C / B→C | lazy cross + WCCN | cross A→B top-1 | same proc↔proc (A→B) mean / max | diff proc↔proc mean |
|---|---|---|---|---|---|---|---|
| dsp_natural | 2.8 % [1-14] | 2.6 % [0-7] | 2.6 % [0-6] / 2.6 % [0-9] / 2.6 % [0-9] | 2.6 % [0-8] | 96 % | 0.541 / 0.773 | 0.039 |
| dsp_balanced | 12.9 % [5-21] | 2.7 % [0-9] | 2.6 % [0-8] / 3.8 % [0-9] / 2.6 % [0-7] | 2.6 % [0-9] | 96 % | 0.543 / 0.730 | 0.067 |
| dsp_strong | 22.0 % [15-37] | 0.2 % [0-2] | 1.2 % [0-4] / 1.3 % [0-4] / 1.4 % [0-4] | 1.3 % [0-3] | 96 % | 0.539 / 0.683 | 0.077 |
| psn_world | 2.6 % [0-12] | 2.6 % [0-9] | 3.7 % [0-9] / 2.6 % [0-9] / 2.6 % [0-9] | 2.6 % [0-9] | 93 % | 0.525 / 0.755 | 0.093 |
| psn_world_cmvn | 3.8 % [0-16] | 5.1 % [0-14] | 5.0 % [1-14] / 5.1 % [0-16] / 5.1 % [0-16] | 3.8 % [1-14] | 93 % | 0.505 / 0.750 | 0.099 |
| knnvc_pseudo | 29.5 % [22-41] | 30.4 % [24-46] | 34.6 % [28-40] / 33.3 % [23-39] / 34.7 % [23-41] | 35.9 % [27-41] | 41 % | 0.415 / 0.665 | 0.323 |

VAD failures (no speech detected by the GE2E front-end) for psn_world_cmvn: ['psn_world_cmvn/A/pyannote-sample-speaker90__0.wav', 'psn_world_cmvn/A/pyannote-sample-speaker90__1.wav', 'psn_world_cmvn/A/pyannote-sample-speaker91__0.wav', 'psn_world_cmvn/B/pyannote-sample-speaker90__0.wav', 'psn_world_cmvn/B/pyannote-sample-speaker90__1.wav', 'psn_world_cmvn/B/pyannote-sample-speaker91__0.wav', 'psn_world_cmvn/B/pyannote-sample-speaker91__1.wav', 'psn_world_cmvn/B/pyannote-sample-speaker91__2.wav', 'psn_world_cmvn/C/pyannote-sample-speaker90__0.wav', 'psn_world_cmvn/C/pyannote-sample-speaker90__1.wav', 'psn_world_cmvn/C/pyannote-sample-speaker91__0.wav', 'psn_world_cmvn/C/pyannote-sample-speaker91__2.wav']

| system | ESTOI | rel. WER | DNSMOS OVRL | DNSMOS SIG | duration ratio | clipped | speech dropouts | RTF (host) | peak RSS child |
|---|---|---|---|---|---|---|---|---|---|
| dsp_natural | 0.76 | 55 % | 2.58 | 2.92 | 1.000 | 0 | 0.1 % | 0.010 | 948 MB |
| dsp_balanced | 0.64 | 60 % | 2.59 | 2.93 | 1.000 | 0 | 0.1 % | 0.010 | 1167 MB |
| dsp_strong | 0.54 | 62 % | 2.56 | 2.90 | 1.000 | 0 | 0.1 % | 0.010 | 1226 MB |
| psn_world | 0.69 | 40 % | 2.70 | 3.12 | 1.000 | 0 | 0.0 % | 0.232 | 1226 MB |
| psn_world_cmvn | 0.66 | 51 % | 2.59 | 2.99 | 1.000 | 0 | 7.2 % | 0.215 | 1226 MB |
| knnvc_pseudo | 0.45 | 63 % | 2.61 | 3.03 | 0.994 | 0 | 0.0 % | 3.126 | 3025 MB |

## vpc_b3 (separate run)

### Evaluator: ge2e  (original->original EER 5.1 % [0-9], top-1 93 %)

| system | ignorant EER | lazy same-session EER | lazy cross-session EER A→B / A→C / B→C | lazy cross + WCCN | cross A→B top-1 | same proc↔proc (A→B) mean / max | diff proc↔proc mean |
|---|---|---|---|---|---|---|---|
| vpc_b3 | 48.7 % [37-58] | 51.3 % [30-62] | 47.4 % [40-55] / 42.3 % [33-47] / 47.4 % [36-57] | 44.9 % [39-54] | 15 % | 0.629 / 0.849 | 0.607 |

### Evaluator: mfcc_stats  (original->original EER 20.0 % [3-28], top-1 74 %)

| system | ignorant EER | lazy same-session EER | lazy cross-session EER A→B / A→C / B→C | lazy cross + WCCN | cross A→B top-1 | same proc↔proc (A→B) mean / max | diff proc↔proc mean |
|---|---|---|---|---|---|---|---|
| vpc_b3 | 50.0 % [46-60] | 51.3 % [37-59] | 50.0 % [46-63] / 51.3 % [45-59] / 47.4 % [40-54] | 50.2 % [44-62] | 15 % | 0.309 / 0.862 | 0.308 |

### Evaluator: vpc_ecapa  (original->original EER 0.6 % [0-4], top-1 100 %)

| system | ignorant EER | lazy same-session EER | lazy cross-session EER A→B / A→C / B→C | lazy cross + WCCN | cross A→B top-1 | same proc↔proc (A→B) mean / max | diff proc↔proc mean |
|---|---|---|---|---|---|---|---|
| vpc_b3 | 43.6 % [33-50] | 43.6 % [30-50] | 45.0 % [37-50] / 41.2 % [34-46] / 42.1 % [35-48] | 44.7 % [32-50] | 26 % | 0.409 / 0.651 | 0.376 |

### Evaluator: resnet_vox1  (original->original EER 0.3 % [0-5], top-1 96 %)

| system | ignorant EER | lazy same-session EER | lazy cross-session EER A→B / A→C / B→C | lazy cross + WCCN | cross A→B top-1 | same proc↔proc (A→B) mean / max | diff proc↔proc mean |
|---|---|---|---|---|---|---|---|
| vpc_b3 | 48.7 % [40-56] | 43.6 % [24-50] | 45.2 % [40-53] / 44.7 % [39-53] / 49.8 % [40-56] | 46.1 % [39-50] | 22 % | 0.233 / 0.548 | 0.192 |


| system | ESTOI | rel. WER | DNSMOS OVRL | DNSMOS SIG | duration ratio | clipped | speech dropouts | RTF (host) | peak RSS child |
|---|---|---|---|---|---|---|---|---|---|
| vpc_b3 | 0.15 | 68 % | 2.97 | 3.37 | 0.956 | 0 | 5.8 % | 0.007 | 944 MB |

## Whisper small.en relative WER (TEST, session A)

| system | relative WER (Whisper small.en) | utterances | reference words |
|---|---|---|---|
| dsp_natural | 8.9 % | 27 | 259 |
| dsp_balanced | 8.1 % | 27 | 259 |
| dsp_strong | 10.0 % | 27 | 259 |
| psn_world | 9.3 % | 27 | 259 |
| psn_world_cmvn | 20.5 % | 27 | 259 |
| vpc_b3 | 44.8 % | 27 | 259 |
| knnvc_pseudo | 34.4 % | 27 | 259 |
