import numpy as np
from algorithm.common.weights import equally_spaced_weights
import torch

device = "auto"
device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device


def evaluate_model(model=None, hour_scope=None, reward_dim=None, granularity=None, eval_env=None):
    assert model is not None, "agent model is not provided"
    assert reward_dim is not None, "reward_dim is not set -- e.g. 2"
    assert granularity is not None, "granularity is not set -- e.g. 100"

    weights = equally_spaced_weights(reward_dim, n=granularity)
    utilities = []
    reward_vecs = []
    for weight in weights:
        bill, comfort = play_a_episode_(model=model, weight=weight, eval_env=eval_env)
        reward_vec = np.array([bill, comfort])
        reward_vecs.append(reward_vec)
        utility = np.dot(reward_vec, weight)
        utilities.append(utility)
    return np.mean(utilities), np.mean(reward_vecs, axis=0)


def play_a_episode_(model, weight, action_seq=None, eval_env=None):
    bill = 0
    comfort = 0
    if action_seq is None:
        terminated = False
        truncated = False
        obs, _ = eval_env.reset()
        # print(f"evaluating: {eval_env.hour_scope}")
        while not terminated and not truncated:
            action = model.act(obs=torch.tensor(obs).float().to(device_), epsilon=0,
                               w=torch.tensor(weight).float().to(device_))
            obs_, v_reward, terminated, truncated, _ = eval_env.step(action)
            obs = obs_
            bill += v_reward[0]
            comfort += v_reward[1]
    else:
        obs, _ = eval_env.reset(hour_scope=[0, len(action_seq)])
        for action in action_seq:
            _, v_reward, _, _, _ = eval_env.step(action)
            bill += v_reward[0]
            comfort += v_reward[1]
    return bill, comfort
