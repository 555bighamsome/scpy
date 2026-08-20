from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from jax import lax
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]


def find_data_directory(root: Path = ROOT) -> Path:
    """Find either the workshop clone layout or this public GitHub layout."""
    candidates = (root / "g2ds8-osfstorage-archive", root / "azulejos_data")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Azulejos CSV directory not found. Searched: {searched}")


DATA_DIR = find_data_directory()
WORK_DIR = ROOT / ".analysis_work"
CACHE_DIR = WORK_DIR / "cache"
RESULT_DIR = ROOT / "results" / "change_mechanism_study"
SUBSET_MANIFEST = Path(__file__).with_name("workshop_subset_files.txt")

CHANGE_TYPES = ("probs_type", "points_type", "n_states")
DISPLAY_NAMES = {
    "probs_type": "Reward probability",
    "points_type": "Reward magnitude",
    "n_states": "State structure",
}
MAX_ACTIONS = 6
MAX_STATES = 30
SEED = 2026


def workshop_data_files(data_dir: Path = DATA_DIR) -> list[Path]:
    """Return the exact 400-file workshop subset used for the reported results."""
    if not SUBSET_MANIFEST.exists():
        return sorted(data_dir.glob("*.csv"))
    names = [line.strip() for line in SUBSET_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    files = [data_dir / name for name in names]
    missing = [path.name for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} workshop CSV files; first missing file: {missing[0]}")
    return files


@dataclass
class Session:
    session_id: str
    participant: str
    task_number: int
    change_type: str
    change_level: str
    n_actions: int
    actions: np.ndarray
    rewards: np.ndarray
    states: np.ndarray
    blocks: np.ndarray


def load_sessions(rebuild: bool = False) -> list[Session]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "change_sessions.pkl"
    if cache_path.exists() and not rebuild:
        with cache_path.open("rb") as stream:
            return pickle.load(stream)

    columns = [
        "participant_id", "trial_index", "trial_epoch", "session", "task_number",
        "state", "arm_chosen", "points_won", "block", "n_actions",
        "block_change_type", "block_change_level",
    ]
    sessions: list[Session] = []
    files = workshop_data_files()
    for file_index, path in enumerate(files, 1):
        try:
            frame = pd.read_csv(path, usecols=columns, low_memory=False)
        except Exception:
            continue
        frame["session"] = pd.to_numeric(frame["session"], errors="coerce")
        frame["task_number"] = pd.to_numeric(frame["task_number"], errors="coerce")
        frame["arm_chosen"] = pd.to_numeric(frame["arm_chosen"], errors="coerce")
        keep = (
            frame["trial_epoch"].eq("learn_choice")
            & frame["session"].gt(0)
            & frame["task_number"].ge(9)
            & frame["arm_chosen"].ge(0)
            & frame["block_change_type"].isin(CHANGE_TYPES)
        )
        frame = frame.loc[keep].copy()
        for session_number, group in frame.groupby("session", sort=False):
            group = group.sort_values("trial_index")
            if group.empty:
                continue
            change_type = str(group["block_change_type"].dropna().iloc[0])
            participant = str(group["participant_id"].dropna().iloc[0])
            task_number = int(group["task_number"].dropna().iloc[0])
            n_actions = int(group["n_actions"].dropna().iloc[0])
            actions = group["arm_chosen"].to_numpy(dtype=np.int32)
            rewards = pd.to_numeric(group["points_won"], errors="coerce").fillna(0).to_numpy(dtype=np.float32) / 100.0
            states = pd.to_numeric(group["state"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)
            blocks = pd.to_numeric(group["block"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)
            if not (2 <= n_actions <= MAX_ACTIONS):
                continue
            if actions.max(initial=0) >= n_actions or states.max(initial=0) >= MAX_STATES:
                continue
            sessions.append(Session(
                session_id=f"{path.stem}:s{int(session_number)}",
                participant=participant,
                task_number=task_number,
                change_type=change_type,
                change_level=str(group["block_change_level"].dropna().iloc[0]),
                n_actions=n_actions,
                actions=actions,
                rewards=np.clip(rewards, 0.0, 1.0),
                states=states,
                blocks=blocks,
            ))
        if file_index % 50 == 0 or file_index == len(files):
            print(f"Loaded {file_index}/{len(files)} files; {len(sessions)} sessions")

    with cache_path.open("wb") as stream:
        pickle.dump(sessions, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return sessions


def participant_folds(sessions: list[Session], n_folds: int, seed: int = SEED) -> list[set[str]]:
    participants = np.array(sorted({s.participant for s in sessions}), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(participants)
    return [set(x.tolist()) for x in np.array_split(participants, n_folds)]


def split_train_validation(sessions: list[Session], seed: int, fraction: float = 0.12):
    participants = np.array(sorted({s.participant for s in sessions}), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(participants)
    n_val = max(1, int(round(fraction * len(participants))))
    val_ids = set(participants[:n_val].tolist())
    return [s for s in sessions if s.participant not in val_ids], [s for s in sessions if s.participant in val_ids]


def make_sequence_batch(sessions: list[Session]) -> dict[str, np.ndarray]:
    if not sessions:
        raise ValueError("No sessions supplied")
    max_time = max(len(s.actions) for s in sessions)
    count = len(sessions)
    actions = np.zeros((count, max_time), dtype=np.int32)
    rewards = np.zeros((count, max_time), dtype=np.float32)
    states = np.zeros((count, max_time), dtype=np.int32)
    mask = np.zeros((count, max_time), dtype=bool)
    n_actions = np.zeros(count, dtype=np.int32)
    random_nll = np.zeros(count, dtype=np.float32)
    for i, session in enumerate(sessions):
        length = len(session.actions)
        actions[i, :length] = session.actions
        rewards[i, :length] = session.rewards
        states[i, :length] = session.states
        mask[i, :length] = True
        n_actions[i] = session.n_actions
        random_nll[i] = length * math.log(session.n_actions)
    return {
        "actions": actions,
        "rewards": rewards,
        "states": states,
        "mask": mask,
        "n_actions": n_actions,
        "random_nll": random_nll,
    }


def _sigmoid(x):
    return jax.nn.sigmoid(x)


def _softplus(x):
    return jax.nn.softplus(x) + 1e-3


def transformed_rl_parameters(theta: np.ndarray | jnp.ndarray, adaptive: bool):
    if adaptive:
        return _sigmoid(theta[0]), _sigmoid(theta[1]), _softplus(theta[2])
    return _sigmoid(theta[0]), _softplus(theta[1])


def rl_session_nll(theta, batch, adaptive: bool):
    actions = jnp.asarray(batch["actions"])
    rewards = jnp.asarray(batch["rewards"])
    states = jnp.asarray(batch["states"])
    valid = jnp.asarray(batch["mask"])
    n_actions = jnp.asarray(batch["n_actions"])
    n_sessions, _ = actions.shape
    session_index = jnp.arange(n_sessions)
    action_index = jnp.arange(MAX_ACTIONS)[None, :]

    if adaptive:
        alpha0, eta, beta = transformed_rl_parameters(theta, adaptive=True)
    else:
        alpha0, beta = transformed_rl_parameters(theta, adaptive=False)
        eta = jnp.asarray(0.0)

    q0 = jnp.full((n_sessions, MAX_STATES, MAX_ACTIONS), 0.5, dtype=jnp.float32)
    alpha_state0 = jnp.full((n_sessions,), alpha0, dtype=jnp.float32)

    time_inputs = tuple(jnp.swapaxes(x, 0, 1) for x in (actions, rewards, states, valid))

    def step(carry, inputs):
        q_values, alpha_state = carry
        action_t, reward_t, state_t, valid_t = inputs
        q_current = q_values[session_index, state_t, :]
        logits = beta * q_current
        logits = jnp.where(action_index < n_actions[:, None], logits, -1e9)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        chosen_log_prob = log_probs[session_index, action_t]
        chosen_q = q_current[session_index, action_t]
        prediction_error = reward_t - chosen_q
        updated_q = chosen_q + alpha_state * prediction_error
        old_q = q_values[session_index, state_t, action_t]
        replacement = jnp.where(valid_t, updated_q, old_q)
        q_values = q_values.at[session_index, state_t, action_t].set(replacement)
        if adaptive:
            next_alpha = (1.0 - eta) * alpha_state + eta * jnp.clip(jnp.abs(prediction_error), 0.0, 1.0)
            alpha_state = jnp.where(valid_t, next_alpha, alpha_state)
        nll_t = jnp.where(valid_t, -chosen_log_prob, 0.0)
        return (q_values, alpha_state), nll_t

    (_, _), nll_time = lax.scan(step, (q0, alpha_state0), time_inputs)
    return jnp.sum(nll_time, axis=0)


def fit_rl_model(sessions: list[Session], adaptive: bool, starts: int = 3) -> tuple[np.ndarray, dict]:
    batch = make_sequence_batch(sessions)

    def total_nll(theta):
        return jnp.sum(rl_session_nll(theta, batch, adaptive))

    value_grad = jax.jit(jax.value_and_grad(total_nll))

    def objective(theta):
        value, gradient = value_grad(jnp.asarray(theta, dtype=jnp.float32))
        return float(value), np.asarray(gradient, dtype=float)

    rng = np.random.default_rng(SEED + (17 if adaptive else 0))
    base = np.array([-1.4, -2.2, 1.8] if adaptive else [-1.4, 1.8], dtype=float)
    candidates = []
    for start in range(starts):
        initial = base if start == 0 else base + rng.normal(0, 0.7, size=len(base))
        result = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": 250, "ftol": 1e-9})
        candidates.append(result)
    result = min(candidates, key=lambda item: item.fun)
    transformed = transformed_rl_parameters(jnp.asarray(result.x), adaptive)
    values = [float(x) for x in transformed]
    names = ["alpha0", "eta", "beta"] if adaptive else ["alpha", "beta"]
    details = dict(zip(names, values))
    details.update({"train_nll": float(result.fun), "success": bool(result.success), "iterations": int(result.nit)})
    return np.asarray(result.x, dtype=float), details


def evaluate_rl(theta: np.ndarray, sessions: list[Session], adaptive: bool) -> tuple[np.ndarray, np.ndarray]:
    batch = make_sequence_batch(sessions)
    nll = np.asarray(jax.jit(rl_session_nll, static_argnums=2)(jnp.asarray(theta), batch, adaptive), dtype=float)
    return nll, batch["random_nll"].astype(float)


def build_gru_batch(sessions: list[Session]) -> dict[str, np.ndarray]:
    base = make_sequence_batch(sessions)
    count, max_time = base["actions"].shape
    input_size = MAX_ACTIONS + 1 + MAX_STATES + MAX_ACTIONS + len(CHANGE_TYPES) + 2
    x = np.zeros((count, max_time, input_size), dtype=np.float32)
    offset_prev_action = 0
    offset_reward = offset_prev_action + MAX_ACTIONS
    offset_state = offset_reward + 1
    offset_available = offset_state + MAX_STATES
    offset_change = offset_available + MAX_ACTIONS
    offset_level = offset_change + len(CHANGE_TYPES)

    for i, session in enumerate(sessions):
        length = len(session.actions)
        if length > 1:
            previous = session.actions[:-1]
            x[i, np.arange(1, length), offset_prev_action + previous] = 1.0
            x[i, 1:length, offset_reward] = session.rewards[:-1]
        x[i, np.arange(length), offset_state + session.states] = 1.0
        x[i, :length, offset_available:offset_available + session.n_actions] = 1.0
        x[i, :length, offset_change + CHANGE_TYPES.index(session.change_type)] = 1.0
        level_index = 1 if session.change_level == "high" else 0
        x[i, :length, offset_level + level_index] = 1.0
    base["x"] = x
    return base


def initialize_gru(input_size: int, hidden_size: int, seed: int):
    rng = np.random.default_rng(seed)
    def weight(shape, fan_in):
        return jnp.asarray(rng.normal(0, 1 / math.sqrt(fan_in), size=shape), dtype=jnp.float32)
    return {
        "Wxz": weight((hidden_size, input_size), input_size),
        "Whz": weight((hidden_size, hidden_size), hidden_size),
        "bz": jnp.zeros(hidden_size),
        "Wxr": weight((hidden_size, input_size), input_size),
        "Whr": weight((hidden_size, hidden_size), hidden_size),
        "br": jnp.zeros(hidden_size),
        "Wxh": weight((hidden_size, input_size), input_size),
        "Whh": weight((hidden_size, hidden_size), hidden_size),
        "bh": jnp.zeros(hidden_size),
        "Wy": weight((MAX_ACTIONS, hidden_size), hidden_size),
        "by": jnp.zeros(MAX_ACTIONS),
    }


def gru_logits(params, x):
    x_time = jnp.swapaxes(x, 0, 1)
    def step(h_previous, x_t):
        z = jax.nn.sigmoid(x_t @ params["Wxz"].T + h_previous @ params["Whz"].T + params["bz"])
        r = jax.nn.sigmoid(x_t @ params["Wxr"].T + h_previous @ params["Whr"].T + params["br"])
        candidate = jnp.tanh(x_t @ params["Wxh"].T + (r * h_previous) @ params["Whh"].T + params["bh"])
        hidden = (1.0 - z) * h_previous + z * candidate
        return hidden, hidden @ params["Wy"].T + params["by"]
    initial = jnp.zeros((x.shape[0], params["bz"].shape[0]), dtype=jnp.float32)
    _, logits_time = lax.scan(step, initial, x_time)
    return jnp.swapaxes(logits_time, 0, 1)


def gru_session_nll(params, batch):
    x = jnp.asarray(batch["x"])
    actions = jnp.asarray(batch["actions"])
    mask = jnp.asarray(batch["mask"])
    n_actions = jnp.asarray(batch["n_actions"])
    logits = gru_logits(params, x)
    available = jnp.arange(MAX_ACTIONS)[None, None, :] < n_actions[:, None, None]
    logits = jnp.where(available, logits, -1e9)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    chosen = jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]
    return -jnp.sum(jnp.where(mask, chosen, 0.0), axis=1)


def fit_gru(train_sessions: list[Session], val_sessions: list[Session], hidden_size: int, epochs: int, seed: int):
    train_batch = build_gru_batch(train_sessions)
    val_batch = build_gru_batch(val_sessions)
    params = initialize_gru(train_batch["x"].shape[-1], hidden_size, seed)
    optimizer = optax.adam(0.005)
    optimizer_state = optimizer.init(params)

    def mean_nll(model_params, batch):
        return jnp.sum(gru_session_nll(model_params, batch)) / jnp.maximum(jnp.sum(jnp.asarray(batch["mask"])), 1)

    value_grad = jax.value_and_grad(mean_nll)
    @jax.jit
    def step(model_params, state):
        loss, gradients = value_grad(model_params, train_batch)
        updates, state = optimizer.update(gradients, state, model_params)
        return optax.apply_updates(model_params, updates), state, loss

    evaluate_val = jax.jit(lambda p: mean_nll(p, val_batch))
    best_params = params
    best_val = float("inf")
    best_epoch = 0
    patience = 15
    history = []
    for epoch in range(epochs):
        params, optimizer_state, train_loss = step(params, optimizer_state)
        val_loss = float(evaluate_val(params))
        history.append((float(train_loss), val_loss))
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_params = jax.tree.map(lambda x: x.copy(), params)
            best_epoch = epoch + 1
        elif epoch + 1 - best_epoch >= patience:
            break
    return best_params, {"history": history, "best_epoch": best_epoch, "best_val_nll_per_trial": best_val, "hidden_size": hidden_size}


def evaluate_gru(params, sessions: list[Session]) -> tuple[np.ndarray, np.ndarray]:
    batch = build_gru_batch(sessions)
    nll = np.asarray(jax.jit(gru_session_nll)(params, batch), dtype=float)
    return nll, batch["random_nll"].astype(float)


def append_evaluation(rows, sessions, nll, random_nll, model, fold):
    for session, session_nll, session_random in zip(sessions, nll, random_nll):
        rows.append({
            "fold": fold,
            "model": model,
            "session_id": session.session_id,
            "participant": session.participant,
            "task_number": session.task_number,
            "change_type": session.change_type,
            "change_level": session.change_level,
            "n_actions": session.n_actions,
            "n_trials": len(session.actions),
            "nll": float(session_nll),
            "random_nll": float(session_random),
        })


def summarize_results(evaluations: pd.DataFrame, parameters: pd.DataFrame):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    evaluations.to_csv(RESULT_DIR / "heldout_session_predictions.csv", index=False)
    parameters.to_csv(RESULT_DIR / "fitted_parameters.csv", index=False)

    summary = evaluations.groupby(["change_type", "model"], sort=False).agg(
        nll=("nll", "sum"), random_nll=("random_nll", "sum"),
        trials=("n_trials", "sum"), sessions=("session_id", "count"),
        participants=("participant", "nunique"),
    ).reset_index()
    summary["nll_per_trial"] = summary["nll"] / summary["trials"]
    summary["pseudo_r2"] = 1 - summary["nll"] / summary["random_nll"]
    summary.to_csv(RESULT_DIR / "heldout_summary.csv", index=False)

    wide = evaluations.pivot_table(index=["participant", "change_type"], columns="model", values=["nll", "n_trials"], aggfunc="sum")
    comparisons = []
    rng = np.random.default_rng(SEED)
    for change_type in CHANGE_TYPES:
        block = wide.xs(change_type, level="change_type").dropna()
        for model_a, model_b in [("RW", "Adaptive RW"), ("RW", "GRU"), ("Adaptive RW", "GRU")]:
            delta = (block[("nll", model_a)] - block[("nll", model_b)]) / block[("n_trials", model_a)]
            values = delta.to_numpy(float)
            boot = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(5000)])
            comparisons.append({
                "change_type": change_type,
                "comparison": f"{model_a} - {model_b}",
                "delta_nll_per_trial": values.mean(),
                "ci_low": np.quantile(boot, 0.025),
                "ci_high": np.quantile(boot, 0.975),
                "participants": len(values),
            })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(RESULT_DIR / "paired_model_comparisons.csv", index=False)

    order = list(CHANGE_TYPES)
    models = ["RW", "Adaptive RW", "GRU"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(order))
    width = 0.24
    colors = {"RW": "#276fbf", "Adaptive RW": "#2a9d8f", "GRU": "#d9912b"}
    for index, model in enumerate(models):
        values = [float(summary[(summary.change_type == c) & (summary.model == model)].nll_per_trial.iloc[0]) for c in order]
        axes[0].bar(x + (index - 1) * width, values, width, label=model, color=colors[model])
    axes[0].set_xticks(x, [DISPLAY_NAMES[c] for c in order], rotation=12)
    axes[0].set_ylabel("Held-out NLL per trial (lower is better)")
    axes[0].set_title("Predictive performance by environmental change")
    axes[0].legend()

    for index, comparison in enumerate(["RW - Adaptive RW", "RW - GRU"]):
        block = comparison_frame[comparison_frame.comparison == comparison].set_index("change_type").loc[order]
        offset = (index - 0.5) * 0.18
        values = block.delta_nll_per_trial.to_numpy()
        low = values - block.ci_low.to_numpy()
        high = block.ci_high.to_numpy() - values
        axes[1].errorbar(x + offset, values, yerr=[low, high], fmt="o", capsize=5, label=comparison)
    axes[1].axhline(0, color="grey", linewidth=1, linestyle="--")
    axes[1].set_xticks(x, [DISPLAY_NAMES[c] for c in order], rotation=12)
    axes[1].set_ylabel("Held-out ΔNLL per trial (positive favors second model)")
    axes[1].set_title("Participant-bootstrap model improvements")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "heldout_model_comparison.png", dpi=180)
    plt.close(fig)

    payload = {
        "design": {
            "participant_folds": int(evaluations.fold.nunique()),
            "sessions": int(evaluations[evaluations.model == "RW"].session_id.nunique()),
            "participants": int(evaluations.participant.nunique()),
            "trials": int(evaluations[evaluations.model == "RW"].n_trials.sum()),
            "change_types": list(CHANGE_TYPES),
        },
        "summary": summary.to_dict(orient="records"),
        "comparisons": comparison_frame.to_dict(orient="records"),
    }
    (RESULT_DIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(summary, comparison_frame, parameters, payload["design"])


def write_report(summary, comparisons, parameters, design):
    def result(change_type, model):
        row = summary[(summary.change_type == change_type) & (summary.model == model)].iloc[0]
        return f"NLL/trial={row.nll_per_trial:.4f}, pseudo-R²={row.pseudo_r2:.3f}"
    def contrast(change_type, name):
        row = comparisons[(comparisons.change_type == change_type) & (comparisons.comparison == name)].iloc[0]
        return f"ΔNLL/trial={row.delta_nll_per_trial:.4f}, 95% CI [{row.ci_low:.4f}, {row.ci_high:.4f}]"

    lines = [
        "# Environmental change mechanisms in human reinforcement learning",
        "",
        "## Research question",
        "",
        "Do humans use different learning mechanisms for changes in reward probability, reward magnitude, and state structure? Can a surprise-driven adaptive learning-rate model explain those differences, or does a GRU still provide better held-out choice prediction?",
        "",
        "## Confirmatory design",
        "",
        f"- {design['participants']} participants, {design['sessions']} task-space sessions, and {design['trials']:,} valid choices.",
        f"- {design['participant_folds']}-fold cross-validation split by participant; no participant appears in training and test data in the same fold.",
        "- Standard RW: fixed learning rate and inverse temperature, with state-specific action values.",
        "- Adaptive RW: learning rate follows unsigned reward prediction error (Pearce-Hall-style surprise tracking).",
        "- GRU: previous action, previous reward, current state, available actions, change type, and change level.",
        "- Primary metric: held-out negative log-likelihood per choice. Participant bootstrap intervals quantify paired model improvements.",
        "",
        "## Held-out results",
        "",
    ]
    for change_type in CHANGE_TYPES:
        lines.extend([
            f"### {DISPLAY_NAMES[change_type]}", "",
            f"- RW: {result(change_type, 'RW')}",
            f"- Adaptive RW: {result(change_type, 'Adaptive RW')}",
            f"- GRU: {result(change_type, 'GRU')}",
            f"- Adaptive improvement: {contrast(change_type, 'RW - Adaptive RW')}",
            f"- GRU improvement over RW: {contrast(change_type, 'RW - GRU')}",
            "",
        ])
    lines.extend([
        "## Interpretation rule", "",
        "- A positive RW - Adaptive RW contrast means surprise-sensitive learning generalizes better than a fixed learning rate.",
        "- A positive RW - GRU contrast means the recurrent model captures predictive structure beyond standard RW.",
        "- If Adaptive RW approaches GRU, a cognitively interpretable change-sensitive mechanism explains much of the recurrent advantage.",
        "- If GRU remains clearly better, longer history or nonlinear latent-state computations are still needed.",
        "",
        "## Fitted parameters", "",
        "```text\n" + parameters.to_string(index=False) + "\n```", "",
        "## Scope and limitations", "",
        "This is a pre-specified analysis of the locally available 400-file workshop subset, not the full 3,177-file Azulejos dataset. The dynamic model is one interpretable account of surprise-sensitive learning, not a unique proof of the underlying cognitive mechanism. Task dimensions remain correlated, and a future full-data analysis should repeat the comparison across multiple GRU random seeds and alternative adaptive-learning formulations.",
    ])
    (RESULT_DIR / "research_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(rebuild=args.rebuild_cache)
    if args.max_participants:
        selected = sorted({s.participant for s in sessions})[:args.max_participants]
        selected = set(selected)
        sessions = [s for s in sessions if s.participant in selected]
    folds = participant_folds(sessions, args.folds)
    evaluation_rows = []
    parameter_rows = []

    for fold_index, test_ids in enumerate(folds):
        print(f"\n=== Fold {fold_index + 1}/{len(folds)} ===")
        test_sessions = [s for s in sessions if s.participant in test_ids]
        outer_train = [s for s in sessions if s.participant not in test_ids]

        for change_type in CHANGE_TYPES:
            train_condition = [s for s in outer_train if s.change_type == change_type]
            test_condition = [s for s in test_sessions if s.change_type == change_type]
            for adaptive, label in [(False, "RW"), (True, "Adaptive RW")]:
                theta, details = fit_rl_model(train_condition, adaptive=adaptive, starts=args.rl_starts)
                nll, random_nll = evaluate_rl(theta, test_condition, adaptive=adaptive)
                append_evaluation(evaluation_rows, test_condition, nll, random_nll, label, fold_index)
                parameter_rows.append({"fold": fold_index, "change_type": change_type, "model": label, **details})
                print(change_type, label, details)

        gru_train, gru_val = split_train_validation(outer_train, seed=SEED + fold_index)
        gru_params, gru_details = fit_gru(gru_train, gru_val, hidden_size=args.hidden_size, epochs=args.gru_epochs, seed=SEED + fold_index)
        nll, random_nll = evaluate_gru(gru_params, test_sessions)
        append_evaluation(evaluation_rows, test_sessions, nll, random_nll, "GRU", fold_index)
        parameter_rows.append({
            "fold": fold_index, "change_type": "all", "model": "GRU",
            "best_epoch": gru_details["best_epoch"], "best_val_nll_per_trial": gru_details["best_val_nll_per_trial"],
            "hidden_size": gru_details["hidden_size"],
        })
        print("GRU", {key: value for key, value in gru_details.items() if key != "history"})

    summarize_results(pd.DataFrame(evaluation_rows), pd.DataFrame(parameter_rows))
    print(f"\nSaved results to {RESULT_DIR}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--gru-epochs", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--rl-starts", type=int, default=3)
    parser.add_argument("--max-participants", type=int, default=0, help="Pilot/debug only; 0 uses all participants")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
