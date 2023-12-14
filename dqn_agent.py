from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from infra import pytorch_util as ptu
from copy import deepcopy


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()
        self.critic_weight_history = []

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # change datatype to float
        # observation = observation.float()

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        if torch.rand(1) < epsilon:
            action = torch.randint(self.num_actions, ())
        else:
            # print(observation.shape, self.observation_shape)
            qa_values: torch.Tensor = self.critic(observation)
            action = qa_values.argmax(dim=-1)
        # ENDTODO

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(student): compute target values
            next_qa_values: torch.Tensor = self.target_critic(next_obs)
            assert next_qa_values.shape == (
                batch_size,
                self.num_actions,
            ), next_qa_values.shape

            if not self.use_double_q:
                # Standard
                next_q_values, _ = next_qa_values.max(dim=-1)
            else:
                # Double-Q
                assert False, "Double-Q should not be running"
                doubleq_next_qa_values: torch.Tensor = self.critic(next_obs)
                doubleq_next_action = doubleq_next_qa_values.argmax(dim=-1)
                next_q_values = torch.gather(
                    next_qa_values, 1, doubleq_next_action.unsqueeze(1)
                ).squeeze(1)

            assert next_q_values.shape == (batch_size,), next_q_values.shape

            target_values: torch.Tensor = reward + self.discount * next_q_values * (
                1 - done.float()
            )
            assert target_values.shape == (batch_size,), target_values.shape
            # ENDTODO

        # TODO(student): train the critic with the target values
        """
        qa_values = ...
        q_values = ... # Compute from the data actions; see torch.gather
        loss = ...

        """
        # Predict Q-values
        qa_values = self.critic(obs)
        assert qa_values.shape == (batch_size, self.num_actions), qa_values.shape

        # Select Q-values for the actions that were actually taken
        q_values = torch.gather(qa_values, 1, action.unsqueeze(1)).squeeze(1)
        assert q_values.shape == (batch_size,), q_values.shape

        # Compute loss
        loss: torch.Tensor = self.critic_loss(q_values, target_values)
        # ENDTODO

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def swap_critic(self, config, replay_buffer, batch_size: int, epochs: int):
        assert config["swap_critic"]
        assert batch_size % 2 == 0, "batch_size must be even"

        new_critic = config["agent_kwargs"]["make_critic"](
            self.observation_shape, self.num_actions
        )

        if config["swap_critic_averaging"] and len(self.critic_weight_history) > 0:
            weights = config["get_swap_critic_avg_weights"](len(self.critic_weight_history))
            assert np.allclose(weights.sum(), 1.0), weights.sum()

            final_state_dict = {}
            for key in self.critic_weight_history[0]:
                final_state_dict[key] = sum([
                    weights[i] * self.critic_weight_history[i][key]
                    for i in range(len(self.critic_weight_history))
                ])

            new_critic.load_state_dict(final_state_dict) 
            
        new_optim = config["agent_kwargs"]["make_optimizer"](new_critic.parameters())
        new_lr = config["agent_kwargs"]["make_lr_schedule"](new_optim)

        # number of targets to train on (obs and next_obs)
        total_target_count = len(replay_buffer) * epochs * 2

        for _ in range(total_target_count // batch_size):
            batch = replay_buffer.sample(batch_size // 2)

            obs = ptu.from_numpy(batch["observations"])
            next_obs = ptu.from_numpy(batch["next_observations"])
            inputs = torch.cat([obs, next_obs], dim=0)
            assert inputs.shape == (batch_size, *self.observation_shape)

            with torch.no_grad():
                targets = self.critic(inputs)
            output = new_critic(inputs)

            loss = self.critic_loss(output, targets)

            new_optim.zero_grad()
            loss.backward()
            new_optim.step()
            new_lr.step()

        # update the critic
        self.critic = new_critic
        self.critic_optimizer = new_optim
        self.lr_scheduler = new_lr
        self.update_target_critic()


    def save_critic_weights(self):
        if self.critic_weight_history is None:
            self.critic_weight_history = []
        
        # append weights to history and freeze them
        self.critic_weight_history.append(deepcopy(self.critic.state_dict()))

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        if step % self.target_update_period == 0:
            self.update_target_critic()
        # ENDTODO

        return critic_stats
