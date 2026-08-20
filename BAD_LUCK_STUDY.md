# Bad Luck or a Changing World?

## How Humans Detect Changes in Reward Probability

### Research question

When a person receives a bad outcome in a multi-armed bandit task, do they treat it as ordinary bad luck, or as evidence that the reward environment has changed? We test two related hypotheses:

1. Behaviour should change gradually after a genuine change in the identity of the best action, because people need several observations to distinguish a new environment from random noise.
2. If people respond to surprise by temporarily increasing how quickly they learn, a surprise-sensitive adaptive Rescorla–Wagner model should predict held-out choices better than a fixed-learning-rate Rescorla–Wagner model.

### Short answer

Participants clearly adapted to genuine changes, but the adaptation was gradual. Their probability of selecting the best action fell from **0.551 before a change** to **0.263 on the first exposure after the best action changed**, then recovered to **0.391 during early recovery** and **0.457 during later recovery**.

However, one loss immediately after a change did not produce reliably more switching than a loss in a stable period. The adaptive learning-rate model also did not reliably outperform the standard fixed-rate model on unseen participants. The most defensible conclusion is therefore:

> People detect a changing reward environment by accumulating evidence across multiple experiences, rather than reacting strongly to one surprising loss.

This is evidence about behaviour and prediction, not proof of a unique cognitive mechanism.

---

## 1. Dataset

The analysis uses the **Azulejos workshop dataset**, in which people repeatedly choose among several actions and receive rewards. Tasks vary in action count, state structure, visibility, and the way the environment changes between blocks.

This study focuses only on sessions where **reward probabilities change between blocks** (`block_change_type == "probs_type"`). It uses the exact **400-file workshop subset** that produced the reported results. The public repository currently contains 419 CSV files, so [`analysis/workshop_subset_files.txt`](analysis/workshop_subset_files.txt) explicitly records the 400 analyzed files and prevents accidental changes to the sample.

### Units of observation

- **Participant:** one human decision maker.
- **Session:** one task instance completed by a participant.
- **Trial:** one choice and its reward outcome.
- **State:** the current context; values are learned separately for different states.
- **Block:** a period during which the reward-generating structure is intended to remain stable.
- **State exposure:** the next time the same state is encountered, even if other states occurred between the two observations.

The final probability-change sample contains:

- **283 participants**
- **475 sessions**
- **93,413 same-state choice transitions**
- **1,660 change events** where the empirically best action changed at a block boundary

---

## 2. Exactly how the code selects data

The selection is implemented in [`analysis/bad_luck_study.py`](analysis/bad_luck_study.py). Only the required columns are loaded, which reduces memory use:

```python
columns = [
    "participant_id", "trial_index", "trial_epoch", "session", "task_number",
    "state", "arm_chosen", "points_won", "arm_outcomes", "block", "n_actions",
    "block_change_type", "block_change_level",
]
frame = pd.read_csv(path, usecols=columns, low_memory=False)
```

Variables needed for comparisons are converted to numeric values. Invalid strings become missing values rather than being interpreted as numbers:

```python
for column in [
    "session", "task_number", "state", "arm_chosen",
    "points_won", "block", "n_actions"
]:
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
```

The core filter is:

```python
keep = (
    frame["trial_epoch"].eq("learn_choice")
    & frame["session"].gt(0)
    & frame["task_number"].ge(9)
    & frame["arm_chosen"].ge(0)
    & frame["block_change_type"].eq("probs_type")
)
frame = frame.loc[keep].copy()
```

Each condition has a specific purpose:

| Code condition | Meaning | Why it is used |
|---|---|---|
| `trial_epoch == "learn_choice"` | Keep only actual learning choices | Removes instructions, displays, feedback-only rows, or other non-choice epochs |
| `session > 0` | Keep formal task sessions | Excludes setup or non-task rows encoded with session 0 or missing values |
| `task_number >= 9` | Keep the organizer's task-space phase | Excludes earlier introductory or practice tasks |
| `arm_chosen >= 0` | Require a valid recorded choice | Removes omissions and placeholder action codes |
| `block_change_type == "probs_type"` | Keep reward-probability changes only | Directly matches the research question |

The code then groups rows by session and sorts them by `trial_index`. It does **not** treat rows from different sessions as one continuous history.

### Additional model-validity checks

The shared session loader in [`analysis/change_mechanism_study.py`](analysis/change_mechanism_study.py) also requires:

- between 2 and 6 actions;
- every chosen action index to be smaller than `n_actions`;
- every state index to be smaller than the model capacity of 30 states;
- rewards converted from points to the interval `[0, 1]` by dividing by 100 and clipping.

These checks prevent invalid indices and ensure that the model sees rewards on a common scale.

---

## 3. How the best action is identified

The question is not simply whether participants repeat their previous action. We first need to identify which action was objectively best in every **block × state** cell.

Each trial contains `arm_outcomes`, a vector of the counterfactual outcomes for all available actions. For each block and state, the code:

1. parses every outcome vector;
2. keeps the first `n_actions` entries;
3. averages each action's outcome across trials in that block-state cell;
4. selects the action with the largest mean.

Equivalent pseudocode is:

```python
for (block, state), trials in session.groupby(["block", "state"]):
    outcome_matrix = stack(parse(trial.arm_outcomes)[:n_actions])
    mean_outcome = outcome_matrix.mean(axis=0)
    best_action[block, state] = argmax(mean_outcome)
```

This is preferable to defining the best action from rewards that the participant personally obtained, because obtained rewards depend on what the participant chose. Nevertheless, cells with few observations can still give a noisy estimate of the best action.

A **genuine decision-relevant change** occurs when the best action for the same state differs between two consecutive blocks:

```python
best_changed = current_block_best != previous_block_best
```

Changes that alter numerical probabilities but leave the identity of the best action unchanged are retained as a separate control condition.

---

## 4. Behavioural analyses

### 4.1 State-specific sequences

Trials are separated by state and then sorted by time. This matters because the action that is good in State A may be bad in State B. The “previous choice” therefore means the participant's previous choice in the **same state**, not necessarily the immediately preceding row of the full session.

For every valid transition, the code calculates:

- `correct`: whether the current action equals the block-state best action;
- `switched`: whether the current action differs from the previous action in the same state;
- `previous_loss`: whether the previous same-state reward was zero or negative;
- `within_block`: how many times this state has appeared in the current block;
- `best_changed`: whether the best action changed at the start of this block.

### 4.2 Three loss contexts

The analysis compares choices following losses in three situations:

1. **Stable-period loss:** the current state exposure is at least the seventh exposure within the block (`within_block >= 7`). The participant has therefore had time to experience the current environment.
2. **After change: best action changed:** the current exposure is 2–5, the immediately preceding same-state outcome was a loss, and the best action differs from the previous block.
3. **After change: best action unchanged:** the same early post-boundary window, but the best action remains the same.

The second and third conditions distinguish a block boundary that changes what the person should do from a change that does not require a different optimal action.

Switch rates are first shown descriptively. For inference, the code averages within each participant and compares paired participant means. A **5,000-sample participant bootstrap** provides 95% confidence intervals. Resampling participants, rather than individual trials, respects dependence among repeated choices from the same person.

### 4.3 Change-aligned learning curve

For each block boundary where the best action changes, the code aligns same-state choices from six exposures before to twelve exposures after the boundary:

```text
-6 ... -1 | 0 | 1 ... 4 | 8 ... 12
 pre-change  first   early     later
```

Here, relative exposure `0` is the **first encounter with that state in the new block**. The implementation prevents a window from crossing another unrelated block boundary.

Four participant-level windows are compared:

- pre-change: relative exposures −6 to −1;
- first post-change: relative exposure 0;
- early recovery: relative exposures 1 to 4;
- later recovery: relative exposures 8 to 12.

---

## 5. Computational models

Both models maintain a value, `Q(s,a)`, for every state-action pair. All values begin at 0.5, so initialization does not use future outcomes.

### 5.1 Fixed Rescorla–Wagner model

After choosing action `a` in state `s` and receiving reward `r`, the prediction error is:

```text
δₜ = rₜ − Qₜ(sₜ,aₜ)
```

The chosen value is updated by:

```text
Qₜ₊₁(sₜ,aₜ) = Qₜ(sₜ,aₜ) + αδₜ
```

`α` is one fixed learning rate between 0 and 1. Choices are generated by a softmax rule:

```text
P(aₜ | sₜ) ∝ exp(βQₜ(sₜ,aₜ))
```

`β > 0` is the inverse temperature: larger values produce more deterministic choices. Actions unavailable in the session are masked before the softmax is calculated.

### 5.2 Adaptive Rescorla–Wagner model

The adaptive model starts with learning rate `α₀`, but allows it to change after prediction errors:

```text
αₜ₊₁ = (1 − η)αₜ + η|δₜ|
```

where `η` controls how rapidly the learning rate follows unsigned surprise. A large unexpected outcome should raise `α`, causing subsequent values to update more quickly.

This is a **Pearce–Hall-style surprise-sensitive model**. It is one possible dynamic-learning-rate model, not a test of every adaptive account. In the present implementation, `αₜ` is one evolving scalar for the session rather than a different learning rate for every state.

### 5.3 Parameter fitting

Parameters are transformed to valid ranges:

- sigmoid transformation for `α`, `α₀`, and `η`, placing them in `(0,1)`;
- softplus transformation for `β`, making it positive.

The code minimizes training negative log-likelihood with L-BFGS-B, using JAX automatic gradients and two starting points per model and fold. The better optimization result is retained.

### 5.4 Held-out prediction

Participants—not trials—are randomly divided into three folds using seed 2026. In each fold:

1. fit both models using participants in the other two folds;
2. freeze the fitted parameters;
3. predict the complete choice sequences of unseen participants;
4. repeat until every participant has appeared in a test fold exactly once.

This avoids the optimistic leakage that would occur if trials from the same participant appeared in both training and test sets.

The primary metric is held-out negative log-likelihood per choice:

```text
NLL per choice = −Σ log P(observed choice) / number of choices
```

Lower is better. The code also reports:

```text
random NLL = number of trials × log(number of available actions)
pseudo-R² = 1 − model NLL / random NLL
```

Model differences are calculated within participant and change level, then given 95% intervals using a 5,000-sample participant bootstrap. The reported difference is:

```text
ΔNLL = RW NLL − Adaptive RW NLL
```

Therefore, a positive value favours Adaptive RW; a negative value favours fixed RW.

---

## 6. Results

### 6.1 Recovery after a genuine change

| Window | Mean best-action choice rate |
|---|---:|
| Before change | 0.551 |
| First exposure after change | 0.263 |
| Early recovery | 0.391 |
| Later recovery | 0.457 |

Participant-bootstrap contrasts:

- First post-change minus pre-change: **−0.288**, 95% CI [−0.333, −0.245], `n = 242`.
- Early recovery minus first post-change: **+0.128**, 95% CI [+0.097, +0.160], `n = 242`.
- Later recovery minus first post-change: **+0.195**, 95% CI [+0.155, +0.233], `n = 239`.

The sharp initial drop confirms that the block change was behaviourally important. The subsequent improvement shows learning, but recovery was not immediate and remained below the pre-change level even at later exposures.

### 6.2 Does one loss trigger a change response?

Pooled descriptive switch rates were:

- stable-period loss: **0.583** across 22,716 transitions;
- loss after a boundary where the best action was unchanged: **0.579** across 2,396 transitions;
- loss after a boundary where the best action changed: **0.624** across 2,951 transitions.

The pooled numbers appear to show more switching after a genuine change, but the paired participant analysis does not confirm a reliable effect:

- best-changed minus stable: **+0.007**, 95% CI [−0.022, +0.036], `n = 236`;
- best-unchanged minus stable: **+0.000**, 95% CI [−0.030, +0.029], `n = 227`.

This difference between pooled and participant-paired summaries is important: participants contribute different numbers of trials, and trial pooling can overweight people or sessions with more observations.

### 6.3 Held-out model prediction

| Change level | Fixed RW NLL/choice | Adaptive RW NLL/choice | RW − Adaptive | 95% CI |
|---|---:|---:|---:|---:|
| Low | 0.8847 | 0.8853 | −0.0006 | [−0.0017, +0.0004] |
| High | 0.8887 | 0.8886 | +0.0002 | [−0.0003, +0.0008] |

Both intervals include zero and the differences are extremely small. There is no reliable held-out advantage for the adaptive model.

The fitted adaptive-rate parameter `η` was also small and unstable across folds: approximately 0.00218, 0.00071, and 0.00003. This means the fitted model behaved increasingly like a fixed-rate learner in some folds.

---

## 7. Interpretation

The behavioural curve answers the first part of the question: participants do respond to real changes in reward probability. They become much less accurate when the optimal action changes, and then gradually recover.

The switch analysis and model comparison refine that conclusion. A single bad outcome does not appear to be a sufficient cue for a qualitatively different response, and a model that directly increases its learning rate after unsigned surprise gives essentially no held-out predictive benefit.

The most interesting mechanism suggested by these results is **evidence accumulation**. A participant may maintain uncertainty about whether the world changed, combine several recent outcomes, and only then revise their action policy. Models with explicit change-point beliefs, Bayesian volatility tracking, or multi-trial surprise histories are therefore stronger follow-up candidates than this single-error adaptive RW.

---

## 8. Limitations

1. **Subset:** the result is based on the 400-file workshop subset, not the complete Azulejos dataset.
2. **Estimated best action:** best actions are inferred from mean counterfactual outcomes within block-state cells; short cells remain noisy.
3. **Task dimensions are correlated:** “high” and “low” change sessions can differ in more than one design property, so this is not a clean randomized causal contrast between all task features.
4. **One adaptive model:** failure of this Pearce–Hall-style implementation does not rule out all dynamic learning-rate or change-detection models.
5. **No GRU in this focused comparison:** GRU utilities exist in the shared code, but the reported “Bad Luck” result compares only fixed and adaptive RW. A GRU would require seed-replicated training and the same participant-held-out evaluation before making a fair claim.
6. **Choice prediction is not mechanism identification:** a model can predict well for reasons that do not match the participant's actual cognitive process.

---

## 9. Reproduction and file map

Install dependencies and run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
PYTHONPATH=analysis pytest -q analysis/test_change_mechanism_study.py
PYTHONPATH=analysis python analysis/bad_luck_study.py
```

Main files:

- [`analysis/bad_luck_study.py`](analysis/bad_luck_study.py): behavioural event construction, bootstrap comparisons, model fitting calls, figures, and output writing.
- [`analysis/change_mechanism_study.py`](analysis/change_mechanism_study.py): session loading, fixed/adaptive RW likelihoods, participant folds, and shared evaluation utilities.
- [`analysis/test_change_mechanism_study.py`](analysis/test_change_mechanism_study.py): integrity tests.
- [`analysis/workshop_subset_files.txt`](analysis/workshop_subset_files.txt): exact 400-file sample manifest.
- [`results/bad_luck_study/`](results/bad_luck_study/): compact published result tables and JSON summaries.
- [`docs/bad-luck.html`](docs/bad-luck.html): public-facing visual research brief.

Large trial-level intermediate tables are generated locally but are intentionally not required to understand the reported conclusions. The compact results in the repository contain the statistics used in this document.
