from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from change_mechanism_study import (
    SEED,
    append_evaluation,
    evaluate_rl,
    fit_rl_model,
    find_data_directory,
    load_sessions,
    participant_folds,
    workshop_data_files,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = find_data_directory(ROOT)
CACHE_DIR = ROOT / ".analysis_work" / "cache"
OUTPUT_DIR = ROOT / "results" / "bad_luck_study"


def parse_vector(value):
    if pd.isna(value):
        return None
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            return None
    array = np.asarray(parsed, dtype=float).reshape(-1)
    return array if len(array) else None


def block_state_best_actions(frame: pd.DataFrame, n_actions: int):
    result = {}
    for (block, state), group in frame.groupby(["block", "state"], sort=False):
        outcomes = [parse_vector(value) for value in group["arm_outcomes"]]
        outcomes = [value[:n_actions] for value in outcomes if value is not None and len(value) >= n_actions]
        if outcomes:
            result[(int(block), int(state))] = int(np.nanargmax(np.nanmean(np.vstack(outcomes), axis=0)))
    return result


def extract_session_events(frame: pd.DataFrame, source_file: str):
    frame = frame.sort_values("trial_index").copy()
    participant = str(frame["participant_id"].dropna().iloc[0])
    session_number = int(frame["session"].dropna().iloc[0])
    task_number = int(frame["task_number"].dropna().iloc[0])
    n_actions = int(frame["n_actions"].dropna().iloc[0])
    level = str(frame["block_change_level"].dropna().iloc[0])
    best_lookup = block_state_best_actions(frame, n_actions)
    event_rows = []
    curve_rows = []

    for state, state_frame in frame.groupby("state", sort=False):
        state_frame = state_frame.sort_values("trial_index").copy()
        records = []
        previous_block_best = None
        block_change_flags = {}
        for block in pd.unique(state_frame["block"]):
            key = (int(block), int(state))
            current_best = best_lookup.get(key)
            if current_best is None:
                continue
            block_change_flags[int(block)] = None if previous_block_best is None else current_best != previous_block_best
            previous_block_best = current_best

        within_counts = {}
        for _, row in state_frame.iterrows():
            block = int(row["block"])
            best_action = best_lookup.get((block, int(state)))
            if best_action is None:
                continue
            within_counts[block] = within_counts.get(block, 0) + 1
            records.append({
                "block": block,
                "within_block": within_counts[block],
                "action": int(row["arm_chosen"]),
                "reward": float(row["points_won"]) / 100.0,
                "best_action": best_action,
                "correct": int(row["arm_chosen"]) == best_action,
                "best_changed": block_change_flags.get(block),
            })

        for index, record in enumerate(records):
            if index == 0:
                continue
            previous = records[index - 1]
            same_block = record["block"] == previous["block"]
            switched = record["action"] != previous["action"]
            previous_loss = previous["reward"] <= 0
            row = {
                "participant": participant,
                "session_id": f"{source_file}:s{session_number}",
                "task_number": task_number,
                "change_level": level,
                "state": int(state),
                "block": record["block"],
                "within_block": record["within_block"],
                "correct": float(record["correct"]),
                "switched": float(switched),
                "previous_loss": bool(previous_loss),
                "best_changed": record["best_changed"],
            }
            if previous_loss and same_block:
                if record["within_block"] >= 7:
                    row["loss_context"] = "Stable-period loss"
                elif 2 <= record["within_block"] <= 5 and record["best_changed"] is True:
                    row["loss_context"] = "After change: best action changed"
                elif 2 <= record["within_block"] <= 5 and record["best_changed"] is False:
                    row["loss_context"] = "After change: best action unchanged"
                else:
                    row["loss_context"] = None
            else:
                row["loss_context"] = None
            event_rows.append(row)

        boundary_indices = [i for i, item in enumerate(records) if item["within_block"] == 1 and item["best_changed"] is True]
        for boundary in boundary_indices:
            for relative in range(-6, 13):
                position = boundary + relative
                if 0 <= position < len(records):
                    item = records[position]
                    # Prevent a curve window from crossing an unrelated block boundary.
                    if relative < 0 and item["block"] == records[boundary]["block"]:
                        continue
                    if relative >= 0 and item["block"] != records[boundary]["block"]:
                        continue
                    curve_rows.append({
                        "participant": participant,
                        "session_id": f"{source_file}:s{session_number}",
                        "change_level": level,
                        "state": int(state),
                        "relative_exposure": relative,
                        "correct": float(item["correct"]),
                    })
    return event_rows, curve_rows


def build_behavior_data(rebuild: bool = False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    event_cache = CACHE_DIR / "bad_luck_events.pkl"
    curve_cache = CACHE_DIR / "bad_luck_curves.pkl"
    if event_cache.exists() and curve_cache.exists() and not rebuild:
        return pd.read_pickle(event_cache), pd.read_pickle(curve_cache)

    columns = [
        "participant_id", "trial_index", "trial_epoch", "session", "task_number",
        "state", "arm_chosen", "points_won", "arm_outcomes", "block", "n_actions",
        "block_change_type", "block_change_level",
    ]
    all_events, all_curves = [], []
    files = workshop_data_files(DATA_DIR)
    for file_index, path in enumerate(files, 1):
        try:
            frame = pd.read_csv(path, usecols=columns, low_memory=False)
        except Exception:
            continue
        for column in ["session", "task_number", "state", "arm_chosen", "points_won", "block", "n_actions"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        keep = (
            frame["trial_epoch"].eq("learn_choice")
            & frame["session"].gt(0)
            & frame["task_number"].ge(9)
            & frame["arm_chosen"].ge(0)
            & frame["block_change_type"].eq("probs_type")
        )
        frame = frame.loc[keep].copy()
        for _, group in frame.groupby("session", sort=False):
            events, curves = extract_session_events(group, path.stem)
            all_events.extend(events)
            all_curves.extend(curves)
        if file_index % 50 == 0 or file_index == len(files):
            print(f"Behavior extraction {file_index}/{len(files)} files: {len(all_events)} transitions")
    events = pd.DataFrame(all_events)
    curves = pd.DataFrame(all_curves)
    events.to_pickle(event_cache)
    curves.to_pickle(curve_cache)
    return events, curves


def participant_bootstrap_difference(frame, category_a, category_b, seed=SEED, samples=5000):
    means = frame.groupby(["participant", "loss_context"])["switched"].mean().unstack()
    paired = means[[category_a, category_b]].dropna()
    differences = (paired[category_a] - paired[category_b]).to_numpy(float)
    rng = np.random.default_rng(seed)
    bootstrap = np.array([rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(samples)])
    return {
        "category_a": category_a,
        "category_b": category_b,
        "difference": float(differences.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "participants": int(len(differences)),
    }


def change_curve_summary(curves, seed=SEED, samples=5000):
    window = np.select(
        [
            curves["relative_exposure"].between(-6, -1),
            curves["relative_exposure"].eq(0),
            curves["relative_exposure"].between(1, 4),
            curves["relative_exposure"].between(8, 12),
        ],
        ["pre_change", "first_post_change", "early_recovery", "late_recovery"],
        default=None,
    )
    data = curves.assign(window=window)
    data = data[data.window.notna()]
    participant_means = data.groupby(["participant", "window"])["correct"].mean().unstack()
    means = {name: float(participant_means[name].mean()) for name in participant_means.columns}
    rng = np.random.default_rng(seed)
    contrasts = []
    for first, second in [
        ("first_post_change", "pre_change"),
        ("early_recovery", "first_post_change"),
        ("late_recovery", "first_post_change"),
    ]:
        difference = (participant_means[first] - participant_means[second]).dropna().to_numpy(float)
        bootstrap = np.array([rng.choice(difference, size=len(difference), replace=True).mean() for _ in range(samples)])
        contrasts.append({
            "contrast": f"{first} - {second}",
            "difference": float(difference.mean()),
            "ci_low": float(np.quantile(bootstrap, 0.025)),
            "ci_high": float(np.quantile(bootstrap, 0.975)),
            "participants": int(len(difference)),
        })
    return {"participant_means": means, "contrasts": contrasts}


def fit_models(probability_sessions, folds=3):
    rows, parameter_rows = [], []
    for fold, test_participants in enumerate(participant_folds(probability_sessions, folds)):
        train = [session for session in probability_sessions if session.participant not in test_participants]
        test = [session for session in probability_sessions if session.participant in test_participants]
        print(f"Model fold {fold + 1}/{folds}: {len(train)} train, {len(test)} test sessions")
        for adaptive, name in [(False, "RW"), (True, "Adaptive RW")]:
            theta, details = fit_rl_model(train, adaptive=adaptive, starts=2)
            nll, random_nll = evaluate_rl(theta, test, adaptive=adaptive)
            append_evaluation(rows, test, nll, random_nll, name, fold)
            parameter_rows.append({"fold": fold, "model": name, **details})
            print(name, details)
    return pd.DataFrame(rows), pd.DataFrame(parameter_rows)


def model_summary(evaluations):
    summary = evaluations.groupby(["change_level", "model"]).agg(
        nll=("nll", "sum"), random_nll=("random_nll", "sum"),
        trials=("n_trials", "sum"), sessions=("session_id", "count"),
    ).reset_index()
    summary["nll_per_trial"] = summary["nll"] / summary["trials"]
    summary["pseudo_r2"] = 1 - summary["nll"] / summary["random_nll"]

    pivot = evaluations.pivot_table(
        index=["participant", "change_level"], columns="model",
        values=["nll", "n_trials"], aggfunc="sum",
    ).dropna()
    comparisons = []
    rng = np.random.default_rng(SEED)
    for level in ["low", "high"]:
        block = pivot.xs(level, level="change_level")
        values = ((block[("nll", "RW")] - block[("nll", "Adaptive RW")]) / block[("n_trials", "RW")]).to_numpy(float)
        bootstrap = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(5000)])
        comparisons.append({
            "change_level": level,
            "delta_nll_per_trial": values.mean(),
            "ci_low": np.quantile(bootstrap, 0.025),
            "ci_high": np.quantile(bootstrap, 0.975),
            "participants": len(values),
        })
    return summary, pd.DataFrame(comparisons)


def make_figure(events, curves, model_results, model_comparisons):
    contexts = [
        "Stable-period loss",
        "After change: best action unchanged",
        "After change: best action changed",
    ]
    labels = ["Stable\n(random loss)", "Change\nbest unchanged", "Change\nbest changed"]
    colors = ["#8a94a6", "#2a9dba", "#d9912b"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    participant_curve = curves.groupby(["participant", "relative_exposure"])["correct"].mean().reset_index()
    curve_stats = participant_curve.groupby("relative_exposure")["correct"].agg(["mean", "sem"]).reset_index()
    axes[0].plot(curve_stats.relative_exposure, curve_stats["mean"], color="#276fbf", marker="o", markersize=3)
    axes[0].fill_between(
        curve_stats.relative_exposure,
        curve_stats["mean"] - curve_stats["sem"],
        curve_stats["mean"] + curve_stats["sem"], alpha=.2, color="#276fbf",
    )
    axes[0].axvline(-0.5, color="#d04a4a", linestyle="--", label="Best-action change")
    axes[0].set_xlabel("State exposures relative to change")
    axes[0].set_ylabel("Best-action choice rate")
    axes[0].set_title("A. Adaptation around genuine changes")
    axes[0].legend(frameon=False)

    switch_data = events[events.loss_context.isin(contexts)].groupby(["participant", "loss_context"])["switched"].mean().reset_index()
    stats = switch_data.groupby("loss_context")["switched"].agg(["mean", "sem"]).reindex(contexts)
    axes[1].bar(np.arange(3), stats["mean"], yerr=stats["sem"], color=colors, capsize=5)
    axes[1].set_xticks(np.arange(3), labels)
    axes[1].set_ylabel("Switch probability after a loss")
    axes[1].set_title("B. Bad luck versus a changing world")

    order = ["low", "high"]
    x = np.arange(2)
    width = .34
    for index, (model, color) in enumerate([("RW", "#276fbf"), ("Adaptive RW", "#2a9d8f")]):
        values = [model_results[(model_results.change_level == level) & (model_results.model == model)].nll_per_trial.iloc[0] for level in order]
        axes[2].bar(x + (index - .5) * width, values, width, label=model, color=color)
    axes[2].set_xticks(x, ["Low change", "High change"])
    axes[2].set_ylabel("Held-out NLL per choice")
    axes[2].set_title("C. Predicting unseen participants")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bad_luck_main_results.png", dpi=190)
    plt.close(fig)


def write_report(events, curves, recovery, behavior_comparisons, model_results, model_comparisons, parameters):
    contexts = ["Stable-period loss", "After change: best action unchanged", "After change: best action changed"]
    switch = events[events.loss_context.isin(contexts)].groupby("loss_context")["switched"].agg(["mean", "count"])
    lines = [
        "# Bad Luck or a Changing World?",
        "",
        "## Research question",
        "",
        "Can humans distinguish isolated negative outcomes from genuine changes in reward probability, and does a surprise-driven adaptive learning-rate model predict their choices better than a fixed-learning-rate Rescorla-Wagner model?",
        "",
        "## Design",
        "",
        f"- Probability-change task sessions: {events.session_id.nunique():,}",
        f"- Participants: {events.participant.nunique():,}",
        f"- Same-state choice transitions: {len(events):,}",
        "- A genuine decision-relevant change is a block boundary where the empirically best counterfactual arm changes within a state.",
        "- Stable losses are losses occurring after at least six exposures in the same block.",
        "- Post-change losses occur in exposures 1-4 after a block transition and are separated by whether the best action actually changed.",
        "- Models are evaluated using participant-level 3-fold held-out prediction.",
        "",
        "## Behavioral results",
        "",
        f"- Mean best-action choice rate before a genuine change: {recovery['participant_means']['pre_change']:.3f}.",
        f"- First exposure after the best action changed: {recovery['participant_means']['first_post_change']:.3f}.",
        f"- Early recovery (exposures 1-4): {recovery['participant_means']['early_recovery']:.3f}.",
        f"- Later recovery (exposures 8-12): {recovery['participant_means']['late_recovery']:.3f}.",
        "",
        "### Change-aligned participant-bootstrap contrasts",
        "",
    ]
    for result in recovery["contrasts"]:
        lines.append(
            f"- {result['contrast']}: {result['difference']:.3f}, 95% CI "
            f"[{result['ci_low']:.3f}, {result['ci_high']:.3f}], n={result['participants']} participants."
        )
    lines.extend(["", "### Switch rates after losses (pooled descriptive values)", ""])
    for context in contexts:
        if context in switch.index:
            lines.append(f"- {context}: switch probability={switch.loc[context, 'mean']:.3f}, transitions={int(switch.loc[context, 'count']):,}.")
    lines.extend(["", "### Paired within-participant switch contrasts", ""])
    for result in behavior_comparisons:
        lines.append(
            f"- {result['category_a']} minus {result['category_b']}: {result['difference']:.3f}, "
            f"95% CI [{result['ci_low']:.3f}, {result['ci_high']:.3f}], n={result['participants']} participants."
        )
    lines.extend(["", "## Held-out model results", ""])
    for _, row in model_results.iterrows():
        lines.append(
            f"- {row.change_level} change / {row.model}: NLL per choice={row.nll_per_trial:.4f}, "
            f"pseudo-R²={row.pseudo_r2:.3f}, sessions={int(row.sessions)}."
        )
    lines.extend(["", "### Adaptive RW improvement over fixed RW", ""])
    for _, row in model_comparisons.iterrows():
        lines.append(
            f"- {row.change_level}: ΔNLL per choice={row.delta_nll_per_trial:.4f}, "
            f"95% CI [{row.ci_low:.4f}, {row.ci_high:.4f}], n={int(row.participants)} participants."
        )
    lines.extend([
        "", "## Model parameters", "", "```text\n" + parameters.to_string(index=False) + "\n```", "",
        "## Interpretation", "",
        "Participants showed a large immediate loss of accuracy when the best action changed and then recovered gradually over repeated state exposures. However, the paired switch analysis found little evidence that one post-change loss triggered more switching than a random stable-period loss. Likewise, the surprise-driven adaptive RW did not reliably improve held-out prediction. Together, these results suggest that adaptation may depend on accumulating evidence over a longer history rather than increasing learning rate after a single unsigned prediction error.",
        "", "## Limitations", "",
        "Best actions are estimated from the counterfactual arm outcomes recorded within each block and state. This is preferable to using the participant's obtained rewards, but short block-state cells can still be noisy. The analysis uses the 400-file workshop subset and should be replicated on the complete Azulejos dataset. The adaptive RW is one Pearce-Hall-style account of surprise-driven learning, not a unique identification of the cognitive mechanism.",
    ])
    (OUTPUT_DIR / "research_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events, curves = build_behavior_data()
    probability_sessions = [session for session in load_sessions() if session.change_type == "probs_type"]
    evaluations, parameters = fit_models(probability_sessions, folds=3)
    model_results, model_comparisons = model_summary(evaluations)

    contexts = [
        "After change: best action changed",
        "After change: best action unchanged",
    ]
    behavior_comparisons = [
        participant_bootstrap_difference(events, context, "Stable-period loss", seed=SEED + index)
        for index, context in enumerate(contexts)
    ]
    recovery = change_curve_summary(curves)

    events.to_csv(OUTPUT_DIR / "behavioral_transitions.csv", index=False)
    curves.to_csv(OUTPUT_DIR / "change_aligned_learning_curve.csv", index=False)
    evaluations.to_csv(OUTPUT_DIR / "heldout_model_predictions.csv", index=False)
    parameters.to_csv(OUTPUT_DIR / "model_parameters.csv", index=False)
    model_results.to_csv(OUTPUT_DIR / "model_summary.csv", index=False)
    model_comparisons.to_csv(OUTPUT_DIR / "model_comparisons.csv", index=False)
    (OUTPUT_DIR / "behavior_comparisons.json").write_text(json.dumps(behavior_comparisons, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "change_recovery.json").write_text(json.dumps(recovery, indent=2), encoding="utf-8")
    make_figure(events, curves, model_results, model_comparisons)
    write_report(events, curves, recovery, behavior_comparisons, model_results, model_comparisons, parameters)
    print(f"Saved study outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
