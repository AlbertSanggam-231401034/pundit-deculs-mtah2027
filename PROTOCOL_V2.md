# Pundit Deculs v2: protocol before new held-out scores

Development matches: J03WMX, J03WN1, J03WOH. Their previous results are known; they are development data.
Reserved additional matches: J03WOY, J03WPY, J03WQQ, J03WR9. No labels from them enter fitting, configuration search, or preprocessing fitting.

Feature revision: identity-invariant team point-cloud dynamics, ball/player geometry, observed-ball history and forward context; identical extractor for IDSSE and tactical ATD. No challenge GT or frame-specific corrections.

Training recipes fixed in advance: clean-only versus clean+label-blind ball-gap and mixed-tracking augmentation. Mirror home/away in training. One fixed histogram-gradient-boosting classifier configuration. Three fixed decoder strengths (0, 0.35, 1.0).

Development selection: leave-one-match-out over the three development matches. Average spellF1 equally over matches and clean/ball-gap/mixed-tracking scenarios. macroF1 breaks ties; no challenge score participates.

Corruption: ball gaps, temporally correlated measurement noise, temporary player dropouts, duplicate detections. Geometry feature extractor must be invariant to within-team slot permutations. Corruption masks do not read labels.

After development selection, refit on the three development matches and freeze model, decoder parameters, feature schema and source SHA256 before evaluating the four reserved matches. Report all four and all scenarios, including regressions. Compare frozen v1; no automatic post-test tuning.

Final challenge inference and packaging only after these checks. Keep v1 unchanged.
