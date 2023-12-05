import numpy as np
import torch
from moviepy.editor import ImageSequenceClip

from infra import utils
from infra.dqn_atari_config import atari_dqn_config


def create_video(paths, max_videos_to_save=2) -> np.ndarray:
    # reshape the rollouts
    videos = [np.transpose(p["image_obs"], [0, 3, 1, 2]) for p in paths]

    # max rollout length
    max_videos_to_save = np.min([max_videos_to_save, len(videos)])
    max_length = videos[0].shape[0]
    for i in range(max_videos_to_save):
        if videos[i].shape[0] > max_length:
            max_length = videos[i].shape[0]

    # pad rollouts to all be same length
    for i in range(max_videos_to_save):
        if videos[i].shape[0] < max_length:
            padding = np.tile(
                [videos[i][-1]], (max_length - videos[i].shape[0], 1, 1, 1)
            )
            videos[i] = np.concatenate([videos[i], padding], 0)

    # log videos to tensorboard event file
    videos = np.stack(videos[:max_videos_to_save], 0)

    return videos


def save_video(tensor: np.ndarray, filename: str, fps: int = 10):
    N, T, C, H, W = tensor.shape

    # Convert the tensor values to a scale of 0-255 if they are not already
    if tensor.max() <= 1:
        tensor = tensor * 255

    # Convert to uint8
    tensor = tensor.astype(np.uint8)

    # Convert the tensor into a list of arrays representing each frame
    # Assuming that the channel order in tensor is [Channels, Height, Width]
    frame_list = []
    for n in range(N):
        for t in range(T):
            frame = tensor[n, t].transpose(
                1, 2, 0
            )  # Convert from [C, H, W] to [H, W, C]
            frame_list.append(frame)

    # Create a video clip and write it to a file
    clip = ImageSequenceClip(frame_list, fps=fps)
    clip.write_videofile(filename, codec="libx264")


def main():
    config = atari_dqn_config("BreakoutNoFrameskip-v4")
    render_env = config["make_env"](render=True)

    agent = torch.load(
        # "./data_gpu/breakout_BreakoutNoFrameskip-v4_05-12-2023_02-50-09_clip10.0/model_chkpts/model_980000.pt"
        "data/breakout_BreakoutNoFrameskip-v4_04-12-2023_23-15-58_clip10.0/model_chkpts/model_30000.pt"
        # "data_gpu/breakout_BreakoutNoFrameskip-v4_05-12-2023_02-50-09_clip10.0/model_chkpts/model_950000.pt"
    )

    NUM_RENDER_TRAJS = 2

    video_trajectories = utils.sample_n_trajectories(
        render_env,
        agent,
        NUM_RENDER_TRAJS,
        render_env.spec.max_episode_steps,
        render=True,
    )

    videos = create_video(video_trajectories)
    save_video(videos, "video.mp4")


if __name__ == "__main__":
    main()
