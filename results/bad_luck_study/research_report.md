# Bad Luck or a Changing World?

## Research question

Can humans distinguish isolated negative outcomes from genuine changes in reward probability, and does a surprise-driven adaptive learning-rate model predict their choices better than a fixed-learning-rate Rescorla-Wagner model?

## Design

- Probability-change task sessions: 475
- Participants: 283
- Same-state choice transitions: 93,413
- A genuine decision-relevant change is a block boundary where the empirically best counterfactual arm changes within a state.
- Stable losses are losses occurring after at least six exposures in the same block.
- Post-change losses occur in exposures 1-4 after a block transition and are separated by whether the best action actually changed.
- Models are evaluated using participant-level 3-fold held-out prediction.

## Behavioral results

- Mean best-action choice rate before a genuine change: 0.551.
- First exposure after the best action changed: 0.263.
- Early recovery (exposures 1-4): 0.391.
- Later recovery (exposures 8-12): 0.457.

### Change-aligned participant-bootstrap contrasts

- first_post_change - pre_change: -0.288, 95% CI [-0.333, -0.245], n=242 participants.
- early_recovery - first_post_change: 0.128, 95% CI [0.097, 0.160], n=242 participants.
- late_recovery - first_post_change: 0.195, 95% CI [0.155, 0.233], n=239 participants.

### Switch rates after losses (pooled descriptive values)

- Stable-period loss: switch probability=0.583, transitions=22,716.
- After change: best action unchanged: switch probability=0.579, transitions=2,396.
- After change: best action changed: switch probability=0.624, transitions=2,951.

### Paired within-participant switch contrasts

- After change: best action changed minus Stable-period loss: 0.007, 95% CI [-0.022, 0.036], n=236 participants.
- After change: best action unchanged minus Stable-period loss: 0.000, 95% CI [-0.030, 0.029], n=227 participants.

## Held-out model results

- high change / Adaptive RW: NLL per choice=0.8886, pseudo-R²=0.139, sessions=226.
- high change / RW: NLL per choice=0.8887, pseudo-R²=0.139, sessions=226.
- low change / Adaptive RW: NLL per choice=0.8853, pseudo-R²=0.145, sessions=249.
- low change / RW: NLL per choice=0.8847, pseudo-R²=0.146, sessions=249.

### Adaptive RW improvement over fixed RW

- low: ΔNLL per choice=-0.0006, 95% CI [-0.0017, 0.0004], n=192 participants.
- high: ΔNLL per choice=0.0002, 95% CI [-0.0003, 0.0008], n=177 participants.

## Model parameters

```text
 fold       model    alpha     beta    train_nll  success  iterations   alpha0      eta
    0          RW 0.690145 2.988235 55871.539062     True           7      NaN      NaN
    0 Adaptive RW      NaN 2.951303 55846.539062     True          29 0.794024 0.002179
    1          RW 0.655663 3.200322 56808.480469     True           8      NaN      NaN
    1 Adaptive RW      NaN 3.186685 56805.726562     True          26 0.686132 0.000705
    2          RW 0.648651 3.165844 54774.734375     True           8      NaN      NaN
    2 Adaptive RW      NaN 3.165197 54774.742188     True          35 0.649604 0.000033
```

## Interpretation

Participants showed a large immediate loss of accuracy when the best action changed and then recovered gradually over repeated state exposures. However, the paired switch analysis found little evidence that one post-change loss triggered more switching than a random stable-period loss. Likewise, the surprise-driven adaptive RW did not reliably improve held-out prediction. Together, these results suggest that adaptation may depend on accumulating evidence over a longer history rather than increasing learning rate after a single unsigned prediction error.

## Limitations

Best actions are estimated from the counterfactual arm outcomes recorded within each block and state. This is preferable to using the participant's obtained rewards, but short block-state cells can still be noisy. The analysis uses the 400-file workshop subset and should be replicated on the complete Azulejos dataset. The adaptive RW is one Pearce-Hall-style account of surprise-driven learning, not a unique identification of the cognitive mechanism.