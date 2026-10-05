TEST: 9 unseen speakers, 27 utterances. Split: {'train': ['ami-FEE078', 'ami-FEE087', 'arctic-aew-male', 'mssnsd-clnsp1-male'], 'val': ['ami-MEE009', 'pyannote-sample-speaker91'], 'test': ['ami-FEE083', 'ami-FEO070', 'ami-MEE012', 'ami-MEE075', 'ami-MÉO069', 'arctic-a0007-male', 'arctic-axb-female', 'pyannote-sample-speaker90', 'speechbrain-example1']}

### Evaluator: ge2e  (original->original EER 5.1 % [0-9], top-1 93 %)

| system | ignorant EER | lazy same-session EER | lazy cross-session EER A→B / A→C / B→C | lazy cross + WCCN | cross A→B top-1 | same proc↔proc (A→B) mean / max | diff proc↔proc mean |
|---|---|---|---|---|---|---|---|
| dsp_natural | 7.7 % [4-17] | 7.7 % [0-13] | 6.3 % [0-11] / 6.4 % [1-12] / 6.1 % [0-11] | 6.4 % [0-12] | 93 % | 0.820 / 0.942 | 0.588 |
| dsp_balanced | 11.6 % [7-23] | 5.1 % [0-11] | 4.2 % [0-11] / 5.1 % [1-11] / 5.1 % [1-12] | 5.0 % [0-11] | 93 % | 0.821 / 0.919 | 0.585 |
| dsp_strong | 17.9 % [11-31] | 4.5 % [0-14] | 3.8 % [0-12] / 3.8 % [1-13] / 5.1 % [1-12] | 4.2 % [0-12] | 93 % | 0.818 / 0.922 | 0.586 |
| psn_world | 11.5 % [6-20] | 10.3 % [0-19] | 14.1 % [4-22] / 14.4 % [5-23] / 11.5 % [4-20] | 13.1 % [4-22] | 81 % | 0.772 / 0.892 | 0.596 |
| psn_world_cmvn | 11.8 % [7-29] | 13.1 % [1-21] | 14.2 % [4-20] / 15.4 % [5-21] / 12.8 % [3-21] | 14.4 % [4-20] | 85 % | 0.774 / 0.992 | 0.578 |

### Evaluator: mfcc_stats  (original->original EER 20.0 % [3-28], top-1 74 %)

| system | ignorant EER | lazy same-session EER | lazy cross-session EER A→B / A→C / B→C | lazy cross + WCCN | cross A→B top-1 | same proc↔proc (A→B) mean / max | diff proc↔proc mean |
|---|---|---|---|---|---|---|---|
| dsp_natural | 28.4 % [18-35] | 28.2 % [13-40] | 28.2 % [14-39] / 28.1 % [14-37] / 28.2 % [15-36] | 26.7 % [16-33] | 63 % | 0.388 / 0.713 | 0.168 |
| dsp_balanced | 34.7 % [24-44] | 25.6 % [14-37] | 26.6 % [14-37] / 25.3 % [14-38] / 25.6 % [15-37] | 24.4 % [14-33] | 74 % | 0.399 / 0.773 | 0.174 |
| dsp_strong | 40.7 % [30-56] | 28.2 % [11-39] | 27.0 % [10-39] / 25.6 % [11-37] / 25.9 % [11-38] | 28.2 % [11-35] | 67 % | 0.406 / 0.771 | 0.180 |
| psn_world | 24.4 % [19-37] | 20.7 % [11-31] | 20.5 % [11-32] / 20.8 % [13-32] / 20.7 % [11-35] | 19.2 % [11-30] | 63 % | 0.381 / 0.668 | 0.098 |
| psn_world_cmvn | 34.6 % [27-46] | 28.5 % [10-37] | 28.4 % [12-38] / 29.6 % [14-39] / 28.4 % [8-38] | 28.2 % [14-38] | 63 % | 0.313 / 0.847 | 0.091 |

VAD failures (no speech detected by the GE2E front-end) for psn_world_cmvn: ['psn_world_cmvn/A/pyannote-sample-speaker90__0.wav', 'psn_world_cmvn/A/pyannote-sample-speaker90__1.wav', 'psn_world_cmvn/A/pyannote-sample-speaker91__0.wav', 'psn_world_cmvn/B/pyannote-sample-speaker90__0.wav', 'psn_world_cmvn/B/pyannote-sample-speaker90__1.wav', 'psn_world_cmvn/B/pyannote-sample-speaker91__0.wav', 'psn_world_cmvn/B/pyannote-sample-speaker91__1.wav', 'psn_world_cmvn/B/pyannote-sample-speaker91__2.wav', 'psn_world_cmvn/C/pyannote-sample-speaker90__0.wav', 'psn_world_cmvn/C/pyannote-sample-speaker90__1.wav', 'psn_world_cmvn/C/pyannote-sample-speaker91__0.wav', 'psn_world_cmvn/C/pyannote-sample-speaker91__2.wav']

| system | ESTOI | rel. WER | DNSMOS OVRL | DNSMOS SIG | duration ratio | clipped | RTF (host) |
|---|---|---|---|---|---|---|---|
| dsp_natural | 0.76 | 55 % | 2.58 | 2.92 | 1.000 | 0 | n/a (reused renders) |
| dsp_balanced | 0.64 | 60 % | 2.59 | 2.93 | 1.000 | 0 | n/a (reused renders) |
| dsp_strong | 0.54 | 62 % | 2.56 | 2.90 | 1.000 | 0 | n/a (reused renders) |
| psn_world | 0.69 | 40 % | 2.70 | 3.12 | 1.000 | 0 | n/a (reused renders) |
| psn_world_cmvn | 0.66 | 51 % | 2.59 | 2.99 | 1.000 | 0 | n/a (reused renders) |
