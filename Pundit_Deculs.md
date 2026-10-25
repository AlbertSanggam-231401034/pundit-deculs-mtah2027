# Pundit Deculs

## External-only possession inference with identity-invariant geometry

Team: **Pundit Deculs**. Method: **IDSSE v2**.

This document describes the algorithm, external evaluation and measured
limitations. It does not claim a hidden challenge score.

### Inputs and alignment

Final inference uses the supplied tactical ATD metadata and raw coordinates.
ATD frame ID minus one maps to the tactical video timeline. Official half
boundaries are handled independently.

Normalized ATD coordinates are converted to metres and centred on the pitch.
The model samples the centre of each five-frame block at 5 Hz. Each predicted
label is repeated over its block on the 25 Hz output timeline. Frames outside
the match blocks are X.

Final inference reads no challenge ground truth, video, audio, tagging,
homography or human-produced event annotations.

### External training labels

Final supervised learning uses three other games from the public DFL/IDSSE
dataset, version 1.

Published provider fields are converted to targets:
BallStatus=0 becomes D. When active, BallPossession=1/2 becomes H/A.
Unknown cases become X and are excluded from fitting.

These are existing provider labels, not annotations created by this team.
The dataset paper describes TRACAB position collection and DFL/Sportec data
production. We did not independently adjudicate the two ball fields and do
not assume that every edge case matches the challenge definition.

Only match-information and position XML files are used. Separate event XML
files are not model inputs.

| Development game | Original frames at 25 Hz | Samples at 5 Hz |
|---|---:|---:|
| J03WMX | 145967 | 29194 |
| J03WN1 | 141561 | 28313 |
| J03WOH | 137214 | 27443 |

Total original development frames: **424742**.

The final fit uses clean, ball-gap and mixed-tracking variants, plus H/A
mirroring. Augmented samples are correlated copies, not independent games.

### Features, model and selection

A three-class HistGradientBoosting model predicts H/A/D from 51 geometric
and temporal features. Inputs include ball motion and visibility,
nearest-player distances, team counts and spread, surrounding observations,
and minimum-cost point-cloud displacement at 0.2 and 1 second.

Player clouds are canonicalized per team, avoiding tracking-slot identity
dependence. Minimum-assignment displacement is a geometric statistic,
not guaranteed physical player velocity.

Offline context uses earlier and later input observations. Future target
labels are never features.

Synthetic augmentation uses label-blind random ball gaps, correlated
coordinate noise, player dropout, duplicates and slot permutations.

Selection uses leave-one-match-out validation on the three development
games, two fixed training recipes and decoder strengths 0, 0.35 and 1.
Mean spellF1 receives equal game/scenario weights; macroF1 breaks ties.

The selected configuration is robust_aug with decoder strength 0.35.
A three-state Viterbi decoder uses transition and initial probabilities
estimated from development labels with H/A symmetry.

The final model is fitted on all three development games and frozen before
evaluating J03WOY, J03WPY, J03WQQ and J03WR9. No refitting or parameter changes
follow the reserved-game scores.

The complete configuration, feature signature, hashes and results accompany
the saved model in evaluation/experiments/idsse_v2.

### Possession semantics

H/A approximate controlled possession; D approximates inactive play.
The nearest team is evidence, not the sole decision rule. Temporal decoding
discourages unsupported switches during loose balls and duels.

There is no explicit referee, foul or legal-restart detector. Moving a ball
during an inactive restart may therefore be mistaken for live play.
The model does not guarantee correct retention of the last controlling team
through every deflection or aerial contest.

### External results

Mean spellF1 across the four reserved games:

| Input condition | Frozen v1 | Selected v2 |
|---|---:|---:|
| Original tracking | 0.4644 | 0.5525 |
| Synthetic random ball gaps | 0.2686 | 0.4988 |
| Synthetic tracking corruption | 0.2434 | 0.4502 |
| Additional fixed 5-second gaps | 0.3710 | 0.5117 |
| Additional fixed 15-second gaps | 0.3520 | 0.4670 |

Both methods receive the same sampled/corrupted observations.
For v1 these are repeated to 25 Hz, rather than using its original full-rate
observations. V1 learned from one external game and v2 from three.
The comparison does not isolate architecture from extra training data.

Fixed-gap tests are additional diagnostics, not a tuning set.
External tracking is cleaner than automatic challenge ATD. Synthetic
corruption cannot cover every real failure. Four reserved games and their
correlated variants do not establish private challenge accuracy.

### Challenge candidate and audit

The candidate contains 199319 rows: H=59930, A=40130, D=58797, X=40462.
All match-block frames have H/A/D; all outside frames are X.

It reports 434 possession spells with a mean duration of 9.222 seconds.
There are 15 H/A spells shorter than one second.
The longest predicted D spans in the two halves last 142.4 and 208.8 seconds.
Their correctness has not been established.

These statistics describe predictions, not truth. They were not used to
hand-correct labels or tune model parameters.

### Development history and excluded assets

Earlier exploratory hybrid/audio/video prototypes used public challenge
labels for fitting or selection. Those prototypes were withdrawn.
The final v2 model, training labels, selection and temporal parameters use
only other-game IDSSE data. Public-GT-derived fitted models and caches are
excluded from the final inference package.

Tactical/broadcast video and audio, tonal cues and frozen torchvision
R3D-18 KINETICS400_V1 features were explored but are not final-model inputs.

Exploratory team-rule audits also used automated weak phase labels from
three SkillCorner open-data games: 2017461, 2016236 and 2010085, with 93,443
weak-labelled samples across the comparisons. These were agreement audits,
not challenge scores. No SkillCorner targets, features or fitted assets
are used by v2.

The supplied game was watched for understanding/debugging. Observations
were not converted into hand labels, frame-specific corrections or
frame-specific decision rules in v2.

### Reproduction

The public code includes frozen weights and pinned dependencies.
Follow README.md and run:

```bat
python pundit_v2.py candidate --data-dir "E:\More Than A Hack 2027 - Initial Challenge - 13"
python evaluation\validate_submission.py evaluation\candidates\idsse_v2\Pundit_Deculs.csv

```

The packager runs a separately copied source tree in a fresh Python process, blocks access to the original project, compares CSV bytes and runs the official validator. This uses the same local interpreter and installed packages, not an independent OS installation. Exact inference uses the included frozen model.

CSV SHA256: 7967963a92f429a0b4c53433e6a418dcd9a34802cfb0958a29acfface96f61af

### Sources and licences

- Bassek, M., Rein, R., Weber, H., and Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. Scientific Data, 12, 195. https://doi.org/10.1038/s41597-025-04505-y
- DFL/IDSSE dataset, version 1: https://doi.org/10.6084/m9.figshare.28196177.v1 . CC BY 4.0. Changes: extraction, sampling, features, augmentation, training and evaluation. No endorsement is implied.
- SkillCorner open data, exploratory audits only: https://github.com/SkillCorner/opendata
- Metrica Sports challenge code: https://github.com/metrica-sports/more-than-a-hack-2027 . The upstream MIT notice is retained. It grants no rights to redistribute the separately licensed match assets.
