"""GPI-PD algorithm."""
import os
import random
from itertools import chain
from typing import Callable, List, Optional, Union

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from algorithm.common.buffer import ReplayBuffer
from algorithm.common.evaluation import (
    log_all_multi_policy_metrics,
    log_episode_info,
    policy_evaluation_mo,
)
from algorithm.common.model_based.probabilistic_ensemble import (
    ProbabilisticEnsemble,
)
from algorithm.common.model_based.utils import ModelEnv, visualize_eval
from algorithm.common.morl_algorithm import MOAgent, MOPolicy
from algorithm.common.networks import (
    NatureCNN,
    get_grad_norm,
    huber,
    layer_init,
    mlp,
    polyak_update,
)
from algorithm.common.prioritized_buffer import PrioritizedReplayBuffer
from algorithm.common.utils import linearly_decaying_value, unique_tol
from algorithm.common.weights import equally_spaced_weights
from algorithm.linear_support import LinearSupport
from experiment.energy_experiment_utils import evaluate_model
from copy import deepcopy
class QNet(nn.Module):
    """Conditioned MO Q network."""

    def __init__(self, obs_shape, action_dim, rew_dim, net_arch, drop_rate=0.01, layer_norm=True):
        """Initialize the net.

        Args:
            obs_shape: The observation shape.
            action_dim: The action dimension.
            rew_dim: The reward dimension.
            net_arch: The network architecture.
            drop_rate: The dropout rate.
            layer_norm: Whether to use layer normalization.
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.phi_dim = rew_dim

        self.weights_features = mlp(rew_dim, -1, net_arch[:1])
        if len(obs_shape) == 1:
            self.state_features = mlp(obs_shape[0], -1, net_arch[:1])
        elif len(obs_shape) > 1:  # Image observation
            self.state_features = NatureCNN(self.obs_shape, features_dim=net_arch[0])
        self.net = mlp(
            net_arch[0], action_dim * rew_dim, net_arch[1:], drop_rate=drop_rate, layer_norm=layer_norm
        )  # 128/128 256 256 256

        self.apply(layer_init)

    def forward(self, obs, w):
        """Forward pass."""
        sf = self.state_features(obs)
        wf = self.weights_features(w)
        q_values = self.net(sf * wf)
        return q_values.view(-1, self.action_dim, self.phi_dim)  # Batch size X Actions X Rewards


class GPIPD(MOPolicy, MOAgent):
    """GPI-PD Algorithm.

    Sample-Efficient Multi-Objective Learning via Generalized Policy Improvement Prioritization
    Lucas N. Alegre, Ana L. C. Bazzan, Diederik M. Roijers, Ann Nowé, Bruno C. da Silva
    AAMAS 2023
    Paper: https://arxiv.org/abs/2301.07784
    """

    def __init__(
            self,
            env,
            learning_rate: float = 3e-4,
            initial_epsilon: float = 0.01,
            final_epsilon: float = 0.01,
            epsilon_decay_steps: int = None,  # None == fixed epsilon
            tau: float = 1.0,
            target_net_update_freq: int = 1000,  # ignored if tau != 1.0
            buffer_size: int = int(1e6),
            net_arch: List = [256, 256, 256, 256],
            num_nets: int = 2,
            batch_size: int = 128,
            learning_starts: int = 100,
            gradient_updates: int = 20,
            gamma: float = 0.99,
            max_grad_norm: Optional[float] = None,
            use_gpi: bool = True,
            dyna: bool = True,
            per: bool = True,
            gpi_pd: bool = True,
            alpha_per: float = 0.6,
            min_priority: float = 0.01,
            drop_rate: float = 0.01,
            layer_norm: bool = True,
            dynamics_normalize_inputs: bool = False,
            dynamics_uncertainty_threshold: float = 1.5,
            dynamics_train_freq: Callable = lambda timestep: 250,
            dynamics_rollout_len: int = 1,
            dynamics_rollout_starts: int = 5000,
            dynamics_rollout_freq: int = 250,
            dynamics_rollout_batch_size: int = 25000,
            dynamics_buffer_size: int = 100000,
            dynamics_net_arch: List = [256, 256, 256],
            dynamics_ensemble_size: int = 5,
            dynamics_num_elites: int = 2,
            real_ratio: float = 0.5,
            project_name: str = "MORL-Baselines",
            experiment_name: str = "GPI-PD",
            wandb_entity: Optional[str] = None,
            log: bool = True,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            file_path=None
    ):
        """Initialize the GPI-PD algorithm.

        Args:
            env: The environment to learn from.
            learning_rate: The learning rate.
            initial_epsilon: The initial epsilon value.
            final_epsilon: The final epsilon value.
            epsilon_decay_steps: The number of steps to decay epsilon.
            tau: The soft update coefficient.
            target_net_update_freq: The target network update frequency.
            buffer_size: The size of the replay buffer.
            net_arch: The network architecture.
            num_nets: The number of networks.
            batch_size: The batch size.
            learning_starts: The number of steps before learning starts.
            gradient_updates: The number of gradient updates per step.
            gamma: The discount factor.
            max_grad_norm: The maximum gradient norm.
            use_gpi: Whether to use GPI.
            dyna: Whether to use Dyna.
            per: Whether to use PER.
            gpi_pd: Whether to use GPI-PD.
            alpha_per: The alpha parameter for PER.
            min_priority: The minimum priority for PER.
            drop_rate: The dropout rate.
            layer_norm: Whether to use layer normalization.
            dynamics_normalize_inputs: Whether to normalize inputs to the dynamics model.
            dynamics_uncertainty_threshold: The uncertainty threshold for the dynamics model.
            dynamics_train_freq: The dynamics model training frequency.
            dynamics_rollout_len: The rollout length for the dynamics model.
            dynamics_rollout_starts: The number of steps before the first rollout.
            dynamics_rollout_freq: The rollout frequency.
            dynamics_rollout_batch_size: The rollout batch size.
            dynamics_buffer_size: The size of the dynamics model buffer.
            dynamics_net_arch: The network architecture for the dynamics model.
            dynamics_ensemble_size: The ensemble size for the dynamics model.
            dynamics_num_elites: The number of elites for the dynamics model.
            real_ratio: The ratio of real transitions to sample.
            project_name: The name of the project.
            experiment_name: The name of the experiment.
            wandb_entity: The name of the wandb entity.
            log: Whether to log.
            seed: The seed for random number generators.
            device: The device to use.
        """
        print(f"gr:{gradient_updates}")
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device=device)
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.tau = tau
        self.target_net_update_freq = target_net_update_freq
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.use_gpi = use_gpi
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates
        self.num_nets = num_nets
        self.drop_rate = drop_rate
        self.layer_norm = layer_norm
        self.mean_utility = []
        # Q-Networks
        self.q_nets = [
            QNet(
                self.observation_shape,
                self.action_dim,
                self.reward_dim,
                net_arch=net_arch,
                drop_rate=drop_rate,
                layer_norm=layer_norm,
            ).to(self.device)
            for _ in range(self.num_nets)
        ]
        self.target_q_nets = [
            QNet(
                self.observation_shape,
                self.action_dim,
                self.reward_dim,
                net_arch=net_arch,
                drop_rate=drop_rate,
                layer_norm=layer_norm,
            ).to(self.device)
            for _ in range(self.num_nets)
        ]
        for q, target_q in zip(self.q_nets, self.target_q_nets):
            target_q.load_state_dict(q.state_dict())
            for param in target_q.parameters():
                param.requires_grad = False
        self.q_optim = optim.Adam(chain(*[net.parameters() for net in self.q_nets]), lr=self.learning_rate)

        # Prioritized experience replay parameters
        self.per = per
        self.gpi_pd = gpi_pd
        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.observation_shape, 1, rew_dim=self.reward_dim, max_size=buffer_size, action_dtype=np.uint8
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.observation_shape, 1, rew_dim=self.reward_dim, max_size=buffer_size, action_dtype=np.uint8
            )
        self.min_priority = min_priority
        self.alpha = alpha_per

        # model-based parameters
        self.dyna = dyna
        self.dynamics_net_arch = dynamics_net_arch
        self.dynamics = None
        self.dynamics_buffer = None
        if self.dyna:
            self.dynamics = ProbabilisticEnsemble(
                input_dim=self.observation_dim + self.action_dim,
                output_dim=self.observation_dim + self.reward_dim,
                arch=self.dynamics_net_arch,
                normalize_inputs=dynamics_normalize_inputs,
                ensemble_size=dynamics_ensemble_size,
                num_elites=dynamics_num_elites,
                device=self.device,
            )
            self.dynamics_buffer = ReplayBuffer(
                self.observation_shape, 1, rew_dim=self.reward_dim, max_size=dynamics_buffer_size, action_dtype=np.uint8
            )
        self.dynamics_train_freq = dynamics_train_freq
        self.dynamics_buffer_size = dynamics_buffer_size
        self.dynamics_normalize_inputs = dynamics_normalize_inputs
        self.dynamics_num_elites = dynamics_num_elites
        self.dynamics_ensemble_size = dynamics_ensemble_size
        self.dynamics_rollout_len = dynamics_rollout_len
        self.dynamics_rollout_starts = dynamics_rollout_starts
        self.dynamics_rollout_freq = dynamics_rollout_freq
        self.dynamics_rollout_batch_size = dynamics_rollout_batch_size
        self.dynamics_uncertainty_threshold = dynamics_uncertainty_threshold
        self.real_ratio = real_ratio
        self.file_path = file_path
        # logging
        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)
        self.experiment_name = experiment_name

    def get_config(self):
        """Return the configuration of the agent."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps:": self.epsilon_decay_steps,
            "batch_size": self.batch_size,
            "per": self.per,
            "gpi_pd": self.gpi_pd,
            "alpha_per": self.alpha,
            "min_priority": self.min_priority,
            "tau": self.tau,
            "num_nets": self.num_nets,
            "clip_grand_norm": self.max_grad_norm,
            "target_net_update_freq": self.target_net_update_freq,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "dynamics_model_arch": self.dynamics_net_arch,
            "gradient_updates": self.gradient_updates,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "dyna": self.dyna,
            "dynamics_rollout_len": self.dynamics_rollout_len,
            "dynamics_uncertainty_threshold": self.dynamics_uncertainty_threshold,
            "dynamics_rollout_starts": self.dynamics_rollout_starts,
            "dynamics_rollout_freq": self.dynamics_rollout_freq,
            "dynamics_rollout_batch_size": self.dynamics_rollout_batch_size,
            "dynamics_buffer_size": self.dynamics_buffer_size,
            "dynamics_normalize_inputs": self.dynamics_normalize_inputs,
            "dynamics_ensemble_size": self.dynamics_ensemble_size,
            "dynamics_num_elites": self.dynamics_num_elites,
            "real_ratio": self.real_ratio,
            "drop_rate": self.drop_rate,
            "layer_norm": self.layer_norm,
            "seed": self.seed,
        }

    def save(self, save_replay_buffer=True, save_dir="weights/", filename=None):
        """Save the model parameters and the replay buffer."""
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        saved_params = {}
        for i, psi_net in enumerate(self.q_nets):
            saved_params[f"psi_net_{i}_state_dict"] = psi_net.state_dict()
        saved_params["psi_nets_optimizer_state_dict"] = self.q_optim.state_dict()
        saved_params["M"] = self.weight_support
        if self.dyna:
            saved_params["dynamics_state_dict"] = self.dynamics.state_dict()
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = self.experiment_name if filename is None else filename
        # th.save(saved_params, save_dir + "/" + filename + ".tar")
        th.save(self.q_nets, save_dir + filename + '.pth')

    def load(self, path, load_replay_buffer=True):
        """Load the model parameters and the replay buffer."""
        params = th.load(path, map_location=self.device)
        for i, (psi_net, target_psi_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            psi_net.load_state_dict(params[f"psi_net_{i}_state_dict"])
            target_psi_net.load_state_dict(params[f"psi_net_{i}_state_dict"])
        self.q_optim.load_state_dict(params["psi_nets_optimizer_state_dict"])
        self.weight_support = params["M"]
        if self.dyna:
            self.dynamics.load_state_dict(params["dynamics_state_dict"])
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def _sample_batch_experiences(self):
        if not self.dyna or self.global_step < self.dynamics_rollout_starts or len(self.dynamics_buffer) == 0:
            return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)
        else:
            num_real_samples = int(self.batch_size * self.real_ratio)  # real_ratio% of real world data
            if self.per:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones, idxes = self.replay_buffer.sample(
                    num_real_samples, to_tensor=True, device=self.device
                )
            else:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = self.replay_buffer.sample(
                    num_real_samples, to_tensor=True, device=self.device
                )
            m_obs, m_actions, m_rewards, m_next_obs, m_dones = self.dynamics_buffer.sample(
                self.batch_size - num_real_samples, to_tensor=True, device=self.device
            )
            experience_tuples = (
                th.cat([s_obs, m_obs], dim=0),
                th.cat([s_actions, m_actions], dim=0),
                th.cat([s_rewards, m_rewards], dim=0),
                th.cat([s_next_obs, m_next_obs], dim=0),
                th.cat([s_dones, m_dones], dim=0),
            )
            if self.per:
                return experience_tuples + (idxes,)
            return experience_tuples

    @th.no_grad()
    def _rollout_dynamics(self, w: th.Tensor):
        # Dyna Planning
        num_times = int(np.ceil(self.dynamics_rollout_batch_size / 10000))
        batch_size = min(self.dynamics_rollout_batch_size, 10000)
        num_added_imagined_transitions = 0
        for iteration in range(num_times):
            obs = self.replay_buffer.sample_obs(batch_size, to_tensor=False)
            model_env = ModelEnv(self.dynamics, self.env.unwrapped.spec.id, rew_dim=len(w))

            for h in range(self.dynamics_rollout_len):
                obs = th.tensor(obs).to(self.device)
                M = th.stack(self.weight_support)
                M = M.unsqueeze(0).repeat(len(obs), 1, 1)
                obs_m = obs.unsqueeze(1).repeat(1, M.size(1), 1)

                psi_values = self.q_nets[0](obs_m, M)
                q_values = th.einsum("r,bar->ba", w, psi_values).view(obs.size(0), len(self.weight_support),
                                                                      self.action_dim)
                max_q, ac = th.max(q_values, dim=2)
                pi = th.argmax(max_q, dim=1)
                actions = ac.gather(1, pi.unsqueeze(1))
                actions_one_hot = F.one_hot(actions, num_classes=self.action_dim).squeeze(1)

                next_obs_pred, r_pred, dones, info = model_env.step(obs, actions_one_hot, deterministic=False)
                uncertainties = info["uncertainty"]
                obs, actions = obs.cpu().numpy(), actions.cpu().numpy()

                for i in range(len(obs)):
                    if uncertainties[i] < self.dynamics_uncertainty_threshold:
                        self.dynamics_buffer.add(obs[i], actions[i], r_pred[i], next_obs_pred[i], dones[i])
                        num_added_imagined_transitions += 1

                nonterm_mask = ~dones.squeeze(-1)
                if nonterm_mask.sum() == 0:
                    break
                obs = next_obs_pred[nonterm_mask]

        if self.log:
            wandb.log(
                {
                    "dynamics/uncertainty_mean": uncertainties.mean(),
                    "dynamics/uncertainty_max": uncertainties.max(),
                    "dynamics/uncertainty_min": uncertainties.min(),
                    "dynamics/model_buffer_size": len(self.dynamics_buffer),
                    "dynamics/imagined_transitions": num_added_imagined_transitions,
                    "global_step": self.global_step,
                },
            )

    def update(self, weight: th.Tensor):
        """Update the parameters of the networks."""
        critic_losses = []
        for g in range(self.gradient_updates if self.global_step >= self.dynamics_rollout_starts else 1):
            if self.per:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones, idxes = self._sample_batch_experiences()
            else:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = self._sample_batch_experiences()

            if len(self.weight_support) > 1:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = (
                    s_obs.repeat(2, *(1 for _ in range(s_obs.dim() - 1))),
                    s_actions.repeat(2, 1),
                    s_rewards.repeat(2, 1),
                    s_next_obs.repeat(2, *(1 for _ in range(s_obs.dim() - 1))),
                    s_dones.repeat(2, 1),
                )
                # Half of the batch uses the given weight vector, the other half uses weights sampled from the support set
                w = th.vstack(
                    [weight for _ in range(s_obs.size(0) // 2)] + random.choices(self.weight_support,
                                                                                 k=s_obs.size(0) // 2)
                )
            else:
                w = weight.repeat(s_obs.size(0), 1)

            if len(self.weight_support) > 5:
                sampled_w = th.stack([weight] + random.sample(self.weight_support, k=4))
            else:
                sampled_w = th.stack(self.weight_support)

            with th.no_grad():
                # Compute min_i Q_i(s', a, w) . w
                next_q_values = th.stack([target_psi_net(s_next_obs, w) for target_psi_net in self.target_q_nets])
                scalarized_next_q_values = th.einsum("nbar,br->nba", next_q_values, w)  # q_i(s', a, w)
                min_inds = th.argmin(scalarized_next_q_values, dim=0)
                min_inds = min_inds.reshape(1, next_q_values.size(1), next_q_values.size(2), 1).expand(
                    1, next_q_values.size(1), next_q_values.size(2), next_q_values.size(3)
                )
                next_q_values = next_q_values.gather(0, min_inds).squeeze(0)

                # Compute max_a Q(s', a, w) . w
                max_q = th.einsum("br,bar->ba", w, next_q_values)
                max_acts = th.argmax(max_q, dim=1)

                q_targets = next_q_values.gather(
                    1, max_acts.long().reshape(-1, 1, 1).expand(next_q_values.size(0), 1, next_q_values.size(2))
                )
                target_q = q_targets.reshape(-1, self.reward_dim)
                target_q = s_rewards + (1 - s_dones) * self.gamma * target_q

                if self.gpi_pd:
                    target_q_envelope, _ = self._envelope_target(s_next_obs, w, sampled_w)
                    target_q_envelope = s_rewards + (1 - s_dones) * self.gamma * target_q_envelope

            losses = []
            td_errors = []
            gtd_errors = []
            for psi_net in self.q_nets:
                psi_value = psi_net(s_obs, w)
                psi_value = psi_value.gather(
                    1, s_actions.long().reshape(-1, 1, 1).expand(psi_value.size(0), 1, psi_value.size(2))
                )
                psi_value = psi_value.reshape(-1, self.reward_dim)

                if self.gpi_pd:
                    gtd_error = psi_value - target_q_envelope

                td_error = psi_value - target_q
                loss = huber(td_error.abs(), min_priority=self.min_priority)
                losses.append(loss)
                if self.gpi_pd:
                    gtd_errors.append(gtd_error.abs())
                if self.per:
                    td_errors.append(td_error.abs())
            critic_loss = (1 / self.num_nets) * sum(losses)

            self.q_optim.zero_grad()
            critic_loss.backward()

            if self.max_grad_norm is not None:
                if self.log and self.global_step % 100 == 0:
                    wandb.log(
                        {
                            "losses/grad_norm": get_grad_norm(self.q_nets[0].parameters()).item(),
                            "global_step": self.global_step,
                        },
                    )
                for psi_net in self.q_nets:
                    th.nn.utils.clip_grad_norm_(psi_net.parameters(), self.max_grad_norm)
            self.q_optim.step()
            critic_losses.append(critic_loss.item())

            if self.per or self.gpi_pd:
                if self.gpi_pd:
                    gtd_error = th.max(th.stack(gtd_errors), dim=0)[0]
                    gtd_error = gtd_error[: len(idxes)].detach()
                    gper = th.einsum("br,br->b", w[: len(idxes)], gtd_error).abs()
                    gpriority = gper.cpu().numpy().flatten()
                    gpriority = gpriority.clip(min=self.min_priority) ** self.alpha

                if self.per:
                    td_error = th.max(th.stack(td_errors), dim=0)[0]
                    td_error = td_error[: len(idxes)].detach()
                    per = th.einsum("br,br->b", w[: len(idxes)], td_error).abs()
                    priority = per.cpu().numpy().flatten()
                    priority = priority.clip(min=self.min_priority) ** self.alpha

                if self.gpi_pd:
                    self.replay_buffer.update_priorities(idxes, gpriority)
                else:
                    self.replay_buffer.update_priorities(idxes, priority)

        if self.tau != 1 or self.global_step % self.target_net_update_freq == 0:
            for psi_net, target_psi_net in zip(self.q_nets, self.target_q_nets):
                polyak_update(psi_net.parameters(), target_psi_net.parameters(), self.tau)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon, self.epsilon_decay_steps, self.global_step, self.learning_starts,
                self.final_epsilon
            )

        if self.log and self.global_step % 100 == 0:
            if self.per:
                wandb.log(
                    {
                        "metrics/mean_priority": np.mean(priority),
                        "metrics/max_priority": np.max(priority),
                        "metrics/mean_td_error_w": per.abs().mean().item(),
                    },
                    commit=False,
                )
            if self.gpi_pd:
                wandb.log(
                    {
                        "metrics/mean_gpriority": np.mean(gpriority),
                        "metrics/max_gpriority": np.max(gpriority),
                        "metrics/mean_gtd_error_w": gper.abs().mean().item(),
                        "metrics/mean_absolute_diff_gtd_td": (gper - per).abs().mean().item(),
                    },
                    commit=False,
                )
            wandb.log(
                {
                    "losses/critic_loss": np.mean(critic_losses),
                    "metrics/epsilon": self.epsilon,
                    "global_step": self.global_step,
                },
            )

    @th.no_grad()
    def gpi_action(self, obs: th.Tensor, w: th.Tensor, return_policy_index=False, include_w=False):
        """Select an action using GPI."""
        if include_w:
            M = th.stack(self.weight_support + [w])
        else:
            M = th.stack(self.weight_support)

        obs_m = obs.repeat(M.size(0), *(1 for _ in range(obs.dim())))
        q_values = self.q_nets[0](obs_m, M)

        scalar_q_values = th.einsum("r,bar->ba", w, q_values)  # q(s,a,w_i) = q(s,a,w_i) . w
        max_q, a = th.max(scalar_q_values, dim=1)
        policy_index = th.argmax(max_q)  # max_i max_a q(s,a,w_i)
        action = a[policy_index].detach().item()

        if return_policy_index:
            return action, policy_index.item()
        return action

    @th.no_grad()
    def eval(self, obs: np.ndarray, w: np.ndarray) -> int:
        """Select an action for the given obs and weight vector."""
        obs = th.as_tensor(obs).float().to(self.device)
        w = th.as_tensor(w).float().to(self.device)
        if self.use_gpi:
            action = self.gpi_action(obs, w, include_w=False)
        else:
            action = self.max_action(obs, w)
        return action

    def act(self, obs: th.Tensor, w: th.Tensor, epsilon=None) -> int:
        if epsilon is not None:
            eps = self.epsilon
            self.epsilon = epsilon

        if self.np_random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            if self.use_gpi:
                action, policy_index = self.gpi_action(obs, w, return_policy_index=True)
                self.police_indices.append(policy_index)
                if epsilon is not None:
                    self.epsilon = eps
                return action
            else:
                if epsilon is not None:
                    self.epsilon = eps
                return self.max_action(obs, w)

    @th.no_grad()
    def max_action(self, obs: th.Tensor, w: th.Tensor) -> int:
        """Select the greedy action."""
        psi = th.min(th.stack([psi_net(obs, w) for psi_net in self.q_nets]), dim=0)[0]
        # psi = self.psi_nets[0](obs, w)
        q = th.einsum("r,bar->ba", w, psi)
        max_act = th.argmax(q, dim=1)
        return max_act.detach().item()

    @th.no_grad()
    def _reset_priorities(self, w: th.Tensor):
        inds = np.arange(self.replay_buffer.size)
        priorities = np.repeat(0.1, self.replay_buffer.size)
        (
            obs_s,
            actions_s,
            rewards_s,
            next_obs_s,
            dones_s,
        ) = self.replay_buffer.get_all_data(to_tensor=False)
        num_batches = int(np.ceil(obs_s.shape[0] / 1000))
        for i in range(num_batches):
            b = i * 1000
            e = min((i + 1) * 1000, obs_s.shape[0])
            obs, actions, rewards, next_obs, dones = obs_s[b:e], actions_s[b:e], rewards_s[b:e], next_obs_s[
                                                                                                 b:e], dones_s[b:e]
            obs, actions, rewards, next_obs, dones = (
                th.tensor(obs).to(self.device),
                th.tensor(actions).to(self.device),
                th.tensor(rewards).to(self.device),
                th.tensor(next_obs).to(self.device),
                th.tensor(dones).to(self.device),
            )
            q_values = self.q_nets[0](obs, w.repeat(obs.size(0), 1))
            q_a = q_values.gather(1, actions.long().reshape(-1, 1, 1).expand(q_values.size(0), 1,
                                                                             q_values.size(2))).squeeze(1)

            if self.gpi_pd:
                max_next_q, _ = self._envelope_target(next_obs, w.repeat(next_obs.size(0), 1),
                                                      th.stack(self.weight_support))
            else:
                next_q_values = self.q_nets[0](next_obs, w.repeat(next_obs.size(0), 1))
                max_q = th.einsum("r,bar->ba", w, next_q_values)
                max_acts = th.argmax(max_q, dim=1)
                q_targets = self.target_q_nets[0](next_obs, w.repeat(next_obs.size(0), 1))
                q_targets = q_targets.gather(
                    1, max_acts.long().reshape(-1, 1, 1).expand(q_targets.size(0), 1, q_targets.size(2))
                )
                max_next_q = q_targets.reshape(-1, self.reward_dim)

            gtderror = th.einsum("r,br->b", w, (rewards + (1 - dones) * self.gamma * max_next_q - q_a)).abs()
            priorities[b:e] = gtderror.clamp(min=self.min_priority).pow(self.alpha).cpu().detach().numpy().flatten()

        self.replay_buffer.update_priorities(inds, priorities)

    @th.no_grad()
    def _envelope_target(self, obs: th.Tensor, w: th.Tensor, sampled_w: th.Tensor):
        W = sampled_w.unsqueeze(0).repeat(obs.size(0), 1, 1)
        next_obs = obs.unsqueeze(1).repeat(1, sampled_w.size(0), 1)

        next_q_target = th.stack(
            [
                target_net(next_obs, W).view(obs.size(0), sampled_w.size(0), self.action_dim, self.reward_dim)
                for target_net in self.target_q_nets
            ]
        )

        q_values = th.einsum("br,nbpar->nbpa", w, next_q_target)
        min_inds = th.argmin(q_values, dim=0)
        min_inds = min_inds.reshape(1, next_q_target.size(1), next_q_target.size(2), next_q_target.size(3), 1).expand(
            1, next_q_target.size(1), next_q_target.size(2), next_q_target.size(3), next_q_target.size(4)
        )
        next_q_target = next_q_target.gather(0, min_inds).squeeze(0)

        q_values = th.einsum("br,bpar->bpa", w, next_q_target)
        max_q, ac = th.max(q_values, dim=2)
        pi = th.argmax(max_q, dim=1)

        max_next_q = next_q_target.gather(
            2,
            ac.unsqueeze(2).unsqueeze(3).expand(next_q_target.size(0), next_q_target.size(1), 1, next_q_target.size(3)),
        ).squeeze(2)
        max_next_q = max_next_q.gather(1,
                                       pi.reshape(-1, 1, 1).expand(max_next_q.size(0), 1, max_next_q.size(2))).squeeze(
            1)
        return max_next_q, next_q_target

    def set_weight_support(self, weight_list: List[np.ndarray]):
        """Set the weight support set."""
        weights_no_repeats = unique_tol(weight_list)
        self.weight_support = [th.tensor(w).float().to(self.device) for w in weights_no_repeats]

    def train_iteration(
            self,
            total_timesteps,
            weight: np.ndarray,
            weight_support: List[np.ndarray],
            change_w_every_episode: bool = True,
            reset_num_timesteps: bool = True,
            eval_env: Optional[gym.Env] = None,
            eval_freq: int = 1000,
            reset_learning_starts: bool = False,
            reward_dim=2,
            granularity=100,
            save_per=100000,
            save_to=None

    ):
        """Train the agent for one iteration.

        Args:
            total_timesteps (int): Number of timesteps to train for
            weight (np.ndarray): Weight vector
            weight_support (List[np.ndarray]): Weight support set
            change_w_every_episode (bool): Whether to change the weight vector at the end of each episode
            reset_num_timesteps (bool): Whether to reset the number of timesteps
            eval_env (Optional[gym.Env]): Environment to evaluate on
            eval_freq (int): Number of timesteps between evaluations
            reset_learning_starts (bool): Whether to reset the learning starts
        """
        # print("===================================================================")
        weight_support = unique_tol(weight_support)  # remove duplicates
        self.set_weight_support(weight_support)
        tensor_w = th.tensor(weight).float().to(self.device)

        self.police_indices = []
        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        if reset_learning_starts:  # Resets epsilon-greedy exploration
            self.learning_starts = self.global_step

        if self.per and len(self.replay_buffer) > 0:
            self._reset_priorities(tensor_w)

        obs, info = self.env.reset()
        episode_steps = 0
        for _ in range(1, total_timesteps + 1):
            # if self.global_step % 1000 == 0:
            # print(f"----global_step:{self.global_step}----")
            self.global_step += 1
            episode_steps += 1
            if self.global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.act(th.as_tensor(obs).float().to(self.device), tensor_w)

            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)

            self.replay_buffer.add(obs, action, vec_reward, next_obs, terminated)

            if self.global_step >= self.learning_starts:
                if self.dyna:
                    if self.global_step % self.dynamics_train_freq(self.global_step) == 0:
                        m_obs, m_actions, m_rewards, m_next_obs, m_dones = self.replay_buffer.get_all_data()
                        one_hot = np.zeros((len(m_obs), self.action_dim))
                        one_hot[np.arange(len(m_obs)), m_actions.astype(int).reshape(len(m_obs))] = 1
                        X = np.hstack((m_obs, one_hot))
                        Y = np.hstack((m_rewards, m_next_obs - m_obs))
                        mean_holdout_loss = self.dynamics.fit(X, Y)
                        if self.log:
                            wandb.log(
                                {"dynamics/mean_holdout_loss": mean_holdout_loss, "global_step": self.global_step},
                            )

                    if self.global_step >= self.dynamics_rollout_starts and self.global_step % self.dynamics_rollout_freq == 0:
                        self._rollout_dynamics(tensor_w)

                self.update(tensor_w)

            # if eval_env is not None and self.log and self.global_step % eval_freq == 0:
            if self.global_step % eval_freq == 0:

                mean_u, mean_rew_v = evaluate_model(model=self,
                                                    reward_dim=reward_dim,
                                                    granularity=granularity,
                                                    eval_env=eval_env)


                print(f"@episode:{self.num_episodes}\t"
                      f"DQN-agent @ mean_u:{mean_u} -- mean[bill,comfort]:{mean_rew_v}\t "
                      f"were issued for 7 days")


            if self.global_step % save_per == 0:
                self.save(save_replay_buffer=False, save_dir=save_to,
                          filename=str(self.global_step))
                print(f"model saved @ step{self.global_step}")

            if terminated or truncated:
                # print(f"truncated:{truncated}\t@{episode_steps}")
                episode_steps = 0
                obs, _ = self.env.reset()
                self.num_episodes += 1

                if self.log and "episode" in info.keys():
                    log_episode_info(info["episode"], np.dot, weight, self.global_step)
                    wandb.log(
                        {"metrics/policy_index": np.array(self.police_indices), "global_step": self.global_step},
                    )
                    self.police_indices = []

                if change_w_every_episode:
                    # eval_weights = equally_spaced_weights(self.reward_dim, n=100)
                    # weight = random.choices(eval_weights, k=len(weight_support))[0]
                    weight = random.choice(weight_support)
                    tensor_w = th.tensor(weight).float().to(self.device)
            else:
                obs = next_obs



    def train(
            self,
            total_timesteps,
            eval_env,
            ref_point: np.ndarray,
            known_pareto_front: Optional[List[np.ndarray]] = None,
            num_eval_weights_for_front: int = 100,
            num_eval_episodes_for_front: int = 5,
            timesteps_per_iter: int = 1000,
            weight_selection_algo: str = "gpi-ls",
            save_to=None,
            eval_freq=1000,
            reward_dim=2,
            granularity=1000,
            save_per=100000,
    ):
        """Train agent.

        Args:
            total_timesteps (int): Number of timesteps to train for.
            eval_env (gym.Env): Environment to evaluate on.
            ref_point (np.ndarray): Reference point for hypervolume calculation.
            known_pareto_front (Optional[List[np.ndarray]]): Optimal Pareto front if known.
            num_eval_weights_for_front: Number of weights to evaluate for the Pareto front.
            num_eval_episodes_for_front: number of episodes to run when evaluating the policy.
            timesteps_per_iter (int): Number of timesteps to train for per iteration.
            weight_selection_algo (str): Weight selection algorithm to use.
        """
        mean_utility = []
        # print("Train Train Train")
        if self.log:
            self.register_additional_config({"ref_point": ref_point.tolist(), "known_front": known_pareto_front})
        max_iter = total_timesteps // timesteps_per_iter
        linear_support = LinearSupport(num_objectives=self.reward_dim,
                                       epsilon=0.0 if weight_selection_algo == "ols" else None,
                                       verbose=False)
        print(f"linear suppor:{linear_support.get_weight_support()}")
        print(f"ccs:{linear_support.ccs}")
        weight_history = []

        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)

        for iter in range(1, max_iter + 1):
            # print(f"iter:{iter}")
            if weight_selection_algo == "ols" or weight_selection_algo == "gpi-ls":
                if weight_selection_algo == "gpi-ls":
                    self.set_weight_support(linear_support.get_weight_support())
                    use_gpi = self.use_gpi
                    self.use_gpi = True
                    w = linear_support.next_weight(
                        algo="gpi-ls", gpi_agent=self, env=eval_env, rep_eval=num_eval_episodes_for_front
                    )
                    self.use_gpi = use_gpi
                else:
                    w = linear_support.next_weight(algo="ols")

                if w is None:
                    break
            else:
                raise ValueError(f"Unknown algorithm {weight_selection_algo}.")

            # print("Next weight vector:", w)
            weight_history.append(w)
            if weight_selection_algo == "gpi-ls":
                M = linear_support.get_weight_support() + linear_support.get_corner_weights(top_k=4) + [w]
            elif weight_selection_algo == "ols":
                M = linear_support.get_weight_support() + [w]
            else:
                M = None

            self.train_iteration(
                total_timesteps=timesteps_per_iter,
                weight=w,
                weight_support=M,
                change_w_every_episode=weight_selection_algo == "gpi-ls",
                eval_env=eval_env,
                eval_freq=eval_freq,
                reset_num_timesteps=False,
                reset_learning_starts=False,
                save_to=save_to,
                reward_dim=reward_dim,
                granularity=granularity,
                save_per=save_per,
            )

            if weight_selection_algo == "ols":
                value = policy_evaluation_mo(self, eval_env, w, rep=num_eval_episodes_for_front)[3]
                linear_support.add_solution(value, w)
            elif weight_selection_algo == "gpi-ls":
                for wcw in M:
                    n_value = policy_evaluation_mo(self, eval_env, wcw, rep=num_eval_episodes_for_front)[3]
                    linear_support.add_solution(n_value, wcw)

        self.save(save_replay_buffer=False, save_dir=save_to,
                  filename=str(self.global_step))

        # np.save(str(self.experiment_name) + ".npy", self.mean_utility)

    def continual_rl_train(self, seed=42, agg_background_power_demands=None, renewable_generations=None, original_model_path=None):
        """
        This function should return the action_sequence of the episode.
        # Anomaly point (week): 4,6,8,10,12,16,23,29,33,38,52
        Returns:
        """

        # wandb.log({'expected utility test': 1,
        #            'expected bill return': 2,
        #            'expected comfort return': 3})

        print("CRL starts")
        anomaly_weeks = [0, 4, 6, 8, 10, 12, 16, 23, 29, 33, 38, 52]

        def week_hour_conversion(week):
            return week * 7 * 24
        def week_day_conversion(week):
            return week*7
        eval_weights = equally_spaced_weights(2, n=100)
        solutions = []
        for i in range(len(eval_weights)):
            solution = np.zeros(2)
            solutions.append(solution)
        print(f"solution len:{len(solutions)}")

        for anomaly_week_idx in range(len(anomaly_weeks)):
            print(f"anomaly_week found @ {anomaly_weeks[anomaly_week_idx]}")
            self.q_nets = th.load(original_model_path, map_location=self.device)
            # self.q_nets = th.load("meta_learning_model/reptile_day_based3_1.pth", map_location=self.device)
            week = anomaly_weeks[anomaly_week_idx]
            # hour_scope = [week_hour_conversion(week), week_hour_conversion(week+1)]
            day_scope = [week_day_conversion(week), week_day_conversion(week)+1]

            env = gym.make('EnergyModel-v0',
                               background_power_demands=agg_background_power_demands,
                               renewable_generations=renewable_generations,
                               renewable_availability=True,
                               task=4,
                               power=1.5,
                               day_scope=day_scope,
                               task_slot=[0, 8])
            self.env = env
            eval_env = gym.make('EnergyModel-v0',
                                    background_power_demands=agg_background_power_demands,
                                    renewable_generations=renewable_generations,
                                    renewable_availability=True,
                                    task=4,
                                    power=1.5,
                                    day_scope=day_scope,
                                    task_slot=[0, 8])
            self.initial_epsilon = 0.7
            self.epsilon_decay_steps = int(96 * 0.8)
            self.train(
                # total_timesteps=24*20,
                total_timesteps=96,
                save_per=100000,
                eval_env=eval_env,
                ref_point=np.array([0.0, 0.0, -200.0]),
                    # known_pareto_front=env.unwrapped.pareto_front(gamma=0.98),
                weight_selection_algo="gpi-ls",
                num_eval_episodes_for_front=1,
                timesteps_per_iter=48,
                save_to="agent_model/" + self.experiment_name + "/" + str(seed) + "/crl_models/_" + str(
                        anomaly_weeks[anomaly_week_idx]) + "/",
                eval_freq=90,
                reward_dim=2,
                granularity=100,
                )
                # get the action sequence of the first 4 weeks
            if anomaly_week_idx < len(anomaly_weeks)-1:
                week_ = anomaly_weeks[anomaly_week_idx]
                week_n = anomaly_weeks[anomaly_week_idx+1]
                day_scope_ = [week_day_conversion(week_), week_day_conversion(week_n)]
                test_env = gym.make('EnergyModel-v0',
                                        background_power_demands=agg_background_power_demands,
                                        renewable_generations=renewable_generations,
                                        renewable_availability=True,
                                        task=4,
                                        power=1.5,
                                        day_scope=day_scope_,
                                        task_slot=[0, 8])
                for i in range(len(eval_weights)):
                    w = eval_weights[i]
                    cumulative_rews = self.play_a_episode_(weight=w,
                                                           eval_env=test_env)
                    solutions[i] += cumulative_rews
        CL_EU = 0
        for i in range(len(eval_weights)):
            w = eval_weights[i]
            print(f"w:{w}\tsolutions:{solutions[i]}")
            CL_EU += np.dot(w, solutions[i])
        print(f"CL_EU:{CL_EU/len(eval_weights)}")
        return solutions

    def train_step(self):
        pass


    def reptile_train(self, agg_background_power_demands,renewable_generations,
                      day_scopes=None,
                      inner_episodes=10,
                      n_iterations=1000,
                      n_task=10,
                      outer_learning_rate=1e-3,
                      save_to=None,
                      save_per = 100):
        eval_weights = equally_spaced_weights(2, n=100)
        weights_before_q_0 = deepcopy(self.q_nets[0].state_dict())
        weights_before_q_1 = deepcopy(self.q_nets[1].state_dict())
        meta_env = gym.make('EnergyModel-v0',
                                   background_power_demands=agg_background_power_demands,
                                   renewable_generations=renewable_generations,
                                   renewable_availability=True,
                                   task=4,
                                   power=1.5,
                                   day_scope=[0,365],
                                   task_slot=[0, 8])
        step = 0
        for iteration in range(n_iterations):
            print('---------------------------Iteration {}------------------------------'.format(iteration))
            # for _ in range(inner_episodes):
            for _ in range(n_task):
                step+=1
                day_scope = random.choice(day_scopes)
                env = gym.make('EnergyModel-v0',
                                   background_power_demands=agg_background_power_demands,
                                   renewable_generations=renewable_generations,
                                   renewable_availability=True,
                                   task=4,
                                   power=1.5,
                                   day_scope=day_scope,
                                   task_slot=[0, 8])
                eval_env = gym.make('EnergyModel-v0',
                                   background_power_demands=agg_background_power_demands,
                                   renewable_generations=renewable_generations,
                                   renewable_availability=True,
                                   task=4,
                                   power=1.5,
                                   day_scope=day_scope,
                                   task_slot=[0, 8])
                print(f"task:{day_scope} is sampled.")
                self.env = env
                print(f"env hour scope:{env.hour_scope}\t eval_env hour scope:{eval_env.hour_scope}")
                self.initial_epsilon = 0.7
                self.epsilon_decay_steps = int(24 * 7 * inner_episodes * 0.8)
                self.train(
                        total_timesteps=24*7*inner_episodes,
                        save_per=100000,
                        eval_env=eval_env,
                        ref_point=np.array([0.0, 0.0, -200.0]),
                        # known_pareto_front=env.unwrapped.pareto_front(gamma=0.98),
                        weight_selection_algo="gpi-ls",
                        num_eval_episodes_for_front=1,
                        timesteps_per_iter=24*7*2,
                        save_to=save_to,
                        eval_freq=24*7,
                        reward_dim=2,
                        granularity=100,
                    )
                # if self.global_step%save_per == 0:
                #     self.save(save_dir=save_to,save_replay_buffer=False,filename=str(step))

            EU = 0
            for w in eval_weights:
                cumulative_rews = self.play_a_episode_(eval_env=meta_env,weight=w)
                print(f"w:{w}\tcumulative_rews:{cumulative_rews}")
                EU += np.dot(w, cumulative_rews)
            mean_EU = EU/len(eval_weights)
            print(f"--------------->>> EU:{mean_EU}")
            postfix = str(int(mean_EU))
            self.save(save_replay_buffer=False, save_dir=save_to+"final/",
                      filename=str(self.seed) + "_" + str(iteration)+"_"+postfix)
            ## Train the model
            weights_after_q_0 = deepcopy(self.q_nets[0].state_dict())
            weights_after_q_1 = deepcopy(self.q_nets[1].state_dict())

            self.q_nets[0].load_state_dict({name: weights_before_q_0[name] + (weights_after_q_0[name] - weights_before_q_0[name]) * outer_learning_rate
                                            for name in weights_before_q_0})

            self.q_nets[1].load_state_dict({name: weights_before_q_1[name] + (weights_after_q_1[name] - weights_before_q_1[name]) * outer_learning_rate
                                            for name in weights_before_q_1})

            print("outer updated done once")


    def get_action_seq(self, hour_scope, w, eval_env):
        action_seq = []
        terminated = False
        truncated = False
        obs, _ = eval_env.reset(hour_scope=hour_scope)

        while not terminated and not truncated:
            action = self.act(obs=obs, w=w, epsilon=0)

            obs_, v_reward, terminated, truncated, _ = eval_env.step(action)
            obs = obs_
            action_seq.append(action)
        return action_seq

    def play_a_episode_(self, eval_env, weight):

        terminated = False
        truncated = False
        steps = 0
        cumulative_rews = np.zeros(2)
        obs, _ = eval_env.reset()
        while not terminated and not truncated:
            steps += 1
            action = self.act(obs=th.as_tensor(obs).float().to(self.device),
                              w=th.as_tensor(weight).float().to(self.device),
                              epsilon=0)

            obs_, rewards, terminated, truncated, _ = eval_env.step(action)
            obs = obs_
            cumulative_rews += rewards

        return cumulative_rews


class GPILS(GPIPD):
    """Model-free version of GPI-PD."""

    def __init__(self, *args, **kwargs):
        """Initialize GPI-LS deactivating the dynamics model."""
        super().__init__(dyna=False, gpi_pd=False, experiment_name="GPI-LS", *args, **kwargs)

