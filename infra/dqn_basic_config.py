import time
from typing import Optional, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from infra import pytorch_util as ptu

from .schedule import ConstantSchedule, LinearSchedule, PiecewiseSchedule


def basic_dqn_config(
    env_name: str,
    exp_name: str,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 1e-3,
    total_steps: int = 300000,
    discount: float = 0.99,
    target_update_period: int = 1000,
    clip_grad_norm: Optional[float] = None,
    use_double_q: bool = False,
    learning_starts: int = 20000,
    batch_size: int = 128,
    weight_decay: bool = False,
    swap_critic: bool = False,
    swap_critic_period: int = 30000,
    swap_critic_init_epochs: int = 3,

    # For the critic averaging experiment
    swap_critic_averaging: bool = False,
    swap_critic_averaging_period: int = 10000,
    **kwargs,
):
    def make_critic(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module:
        return ptu.build_mlp(
            input_size=np.prod(observation_shape),
            output_size=num_actions,
            n_layers=num_layers,
            size=hidden_size,
        )

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        if weight_decay:
            return torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
        else:
            return torch.optim.Adam(params, lr=learning_rate)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1),
            (total_steps * 0.1, 0.02),
        ],
        outside_value=0.02,
    )

    def make_env(render: bool = False):
        return RecordEpisodeStatistics(
            gym.make(env_name, render_mode="rgb_array" if render else None)
        )

    def get_swap_critic_avg_weights(n: int):
        return np.ones(n) / n

    log_string = f"{exp_name}_{env_name}_" + time.strftime("%d-%m-%Y_%H-%M-%S")

    return {
        "agent_kwargs": {
            "make_critic": make_critic,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "target_update_period": target_update_period,
            "clip_grad_norm": clip_grad_norm,
            "use_double_q": use_double_q,
        },
        "exploration_schedule": exploration_schedule,
        "log_name": log_string,
        "make_env": make_env,
        "get_swap_critic_avg_weights": get_swap_critic_avg_weights,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        "weight_decay": weight_decay,
        "swap_critic": swap_critic,
        "swap_critic_period": swap_critic_period,
        "swap_critic_init_epochs": swap_critic_init_epochs,
        "swap_critic_averaging": swap_critic_averaging,
        "swap_critic_averaging_period": swap_critic_averaging_period,
        **kwargs,
    }
