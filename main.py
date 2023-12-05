import argparse
import json
import os

import gym
import numpy as np
import torch
import tqdm

from dqn_agent import DQNAgent
from infra import pytorch_util as ptu
from infra import utils
from infra.dqn_atari_config import atari_dqn_config
from infra.dqn_basic_config import basic_dqn_config
from infra.logger import Logger
from infra.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer

RANDOM_SEED = 0
USE_GPU = True
GPU_ID = 0
LOG_INTERVAL = 1000
EVAL_INTERVAL = 10000
NUM_EVAL_TRAJS = 10
MODEL_SAVE_INTERVAL = 10000


def run_training(config: dict, logger: Logger, data_path: str):
    os.makedirs(os.path.join(data_path, "model_chkpts"))

    env = config["make_env"]()
    eval_env = config["make_env"]()
    exploration_schedule = config["exploration_schedule"]

    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"

    agent = DQNAgent(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )

    ep_len = env.spec.max_episode_steps

    # Replay buffer
    if len(env.observation_space.shape) == 3:
        stacked_frames = True
        frame_history_len = env.observation_space.shape[0]
        assert frame_history_len == 4, "only support 4 stacked frames"
        replay_buffer = MemoryEfficientReplayBuffer(frame_history_len=frame_history_len)
    elif len(env.observation_space.shape) == 1:
        stacked_frames = False
        replay_buffer = ReplayBuffer()
    else:
        raise ValueError(
            f"Unsupported observation space shape: {env.observation_space.shape}"
        )

    observation = None

    def reset_env_training():
        nonlocal observation

        observation = env.reset()

        assert not isinstance(
            observation, tuple
        ), "env.reset() must return np.ndarray - make sure your Gym version uses the old step API"
        observation = np.asarray(observation)

        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            replay_buffer.on_reset(observation=observation[-1, ...])

    reset_env_training()

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=False):
        if step % MODEL_SAVE_INTERVAL == 0:
            torch.save(
                agent, os.path.join(data_path, "model_chkpts", f"model_{step}.pt")
            )

        epsilon = exploration_schedule.value(step)

        # TODO(student): Compute action
        action = agent.get_action(observation, epsilon)

        # TODO(student): Step the environment
        next_observation, reward, done, info = env.step(action)

        next_observation = np.asarray(next_observation)
        truncated = info.get("TimeLimit.truncated", False)

        # TODO(student): Add the data to the replay buffer
        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            # We're using the memory-efficient replay buffer,
            # so we only insert next_observation (not observation)
            replay_buffer.insert(
                action, reward, next_observation[-1], done and not truncated
            )
        else:
            # We're using the regular replay buffer
            replay_buffer.insert(
                observation, action, reward, next_observation, done and not truncated
            )

        # Handle episode termination
        if done:
            reset_env_training()

            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
        else:
            observation = next_observation

        # Main DQN training loop
        if step >= config["learning_starts"]:
            # TODO(student): Sample config["batch_size"] samples from the replay buffer
            # config["batch_size"] is 128 for basic envs and 32 for atari envs
            batch = replay_buffer.sample(config["batch_size"])

            # Convert to PyTorch tensors
            batch = ptu.from_numpy(batch)

            # TODO(student): Train the agent. `batch` is a dictionary of numpy arrays,
            update_info = agent.update(
                batch["observations"],
                batch["actions"],
                batch["rewards"],
                batch["next_observations"],
                batch["dones"],
                step,
            )
            if step % 10000 == 0:
                print("q_values:", update_info["q_values"])
                print("target values:", update_info["target_values"])
                print("critic loss:", update_info["critic_loss"])

            # Logging code
            update_info["epsilon"] = epsilon
            update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]

            if step % LOG_INTERVAL == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                logger.flush()

        if step % EVAL_INTERVAL == 0:
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                eval_env,
                agent,
                NUM_EVAL_TRAJS,
                ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if step % 1000 == 0:
                print(np.mean(returns), "eval_return", step)
                print(np.std(returns), "eval/return_std", step)
                print(np.max(returns), "eval/return_max", step)
                print(np.min(returns), "eval/return_min", step)
                print(np.std(ep_lens), "eval/ep_len_std", step)
                print(np.max(ep_lens), "eval/ep_len_max", step)
                print(np.min(ep_lens), "eval/ep_len_min", step)
                print(len(replay_buffer), "replay_buffer_size", step)

            # if args.num_render_trajectories > 0:
            #     video_trajectories = utils.sample_n_trajectories(
            #         render_env,
            #         agent,
            #         args.num_render_trajectories,
            #         ep_len,
            #         render=True,
            #     )

            #     logger.log_paths_as_videos(
            #         video_trajectories,
            #         step,
            #         fps=fps,
            #         max_videos_to_save=args.num_render_trajectories,
            #         video_title="eval_rollouts",
            #     )


def get_logdir(config: dict, final=False) -> Logger:
    if final:
        data_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "./data_final"
        )
    else:
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./data")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = os.path.join(data_path, config["log_name"])
    return logdir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final", "-f", action="store_true", help="Set this flag to mark as final"
    )

    args = parser.parse_args()
    print("Will save to data_final: ", args.final)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    ptu.init_gpu(use_gpu=USE_GPU, gpu_id=GPU_ID)

    """config = basic_dqn_config(
        "LunarLander-v2",
        exp_name="lunarlander",
        # hidden_size=64,
        # num_layers=2,
        # learning_rate=1e-3,
        # total_steps=300000 * 2,
        # discount=0.99,
        # target_update_period=1000,
        # clip_grad_norm=None,
        # use_double_q=False,
        # learning_starts=20000,
        # batch_size=128,
    )"""

    config = atari_dqn_config(
        "BreakoutNoFrameskip-v4",
        exp_name="breakout",
        use_double_q=True,
        # hidden_size=64,
        # num_layers=2,
        # learning_rate=1.0e-4,
        # total_steps=300000 * 2,
        # discount=0.99,
        # target_update_period=1000,
        # clip_grad_norm=None,
        # learning_starts=20000,
        # batch_size=128,
    )

    data_path = get_logdir(config, final=args.final)

    if os.path.exists(data_path):
        raise Exception("Log directory already exists!")
    else:
        os.makedirs(data_path)

    # save config to log directory
    with open(os.path.join(data_path, "config.json"), "w") as f:
        json.dump(
            config, f, skipkeys=True, indent=4, sort_keys=True, default=lambda e: str(e)
        )

    logger = Logger(os.path.join(data_path, "tensorboard"))

    run_training(config, logger, data_path)
