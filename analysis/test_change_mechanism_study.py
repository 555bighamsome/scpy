import numpy as np

from change_mechanism_study import (
    MAX_ACTIONS,
    Session,
    build_gru_batch,
    evaluate_rl,
    make_sequence_batch,
    transformed_rl_parameters,
)


def example_session(participant="p1", change_type="probs_type"):
    return Session(
        session_id=f"{participant}:s1",
        participant=participant,
        task_number=10,
        change_type=change_type,
        change_level="high",
        n_actions=2,
        actions=np.array([0, 0, 1, 1], dtype=np.int32),
        rewards=np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        states=np.array([0, 0, 0, 0], dtype=np.int32),
        blocks=np.array([0, 0, 1, 1], dtype=np.int32),
    )


def test_sequence_batch_masks_padding_and_random_baseline():
    short = example_session()
    long = example_session("p2")
    long.actions = np.append(long.actions, 0)
    long.rewards = np.append(long.rewards, 1.0)
    long.states = np.append(long.states, 0)
    long.blocks = np.append(long.blocks, 1)
    batch = make_sequence_batch([short, long])
    assert batch["actions"].shape == (2, 5)
    assert batch["mask"].sum(axis=1).tolist() == [4, 5]
    assert np.allclose(batch["random_nll"], [4 * np.log(2), 5 * np.log(2)])


def test_rw_probabilities_are_temporally_valid_and_finite():
    session = example_session()
    theta = np.array([-1.4, 1.8])
    nll, random_nll = evaluate_rl(theta, [session], adaptive=False)
    assert nll.shape == (1,)
    assert np.isfinite(nll[0])
    assert nll[0] > 0
    assert np.isclose(random_nll[0], 4 * np.log(2))


def test_adaptive_parameter_transform_has_valid_ranges():
    alpha0, eta, beta = transformed_rl_parameters(np.array([0.0, 0.0, 0.0]), adaptive=True)
    assert 0 < float(alpha0) < 1
    assert 0 < float(eta) < 1
    assert float(beta) > 0


def test_gru_features_do_not_include_current_choice_or_reward():
    session = example_session()
    batch = build_gru_batch([session])
    x = batch["x"][0]
    assert x.shape[1] > MAX_ACTIONS
    assert np.allclose(x[0, :MAX_ACTIONS], 0)
    assert x[1, session.actions[0]] == 1
    reward_column = MAX_ACTIONS
    assert x[0, reward_column] == 0
    assert x[1, reward_column] == session.rewards[0]
