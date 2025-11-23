import torch
from env.torch_vector_mnk_env import TorchVectorMnkEnv
from selfplay.torch_self_play_wrapper import TorchSelfPlayWrapper


def validate_gpu(agent_policy, opponent_policy, mnk_config, n_episodes=1024, device="cuda"):
    m, n, k = mnk_config

    val_env = TorchVectorMnkEnv(m, n, k, num_envs=n_episodes, device=device)
    wrapper = TorchSelfPlayWrapper(val_env)

    wrapper.set_opponent(opponent_policy)

    agent_sides = torch.zeros(n_episodes, dtype=torch.long, device=device)
    agent_sides[n_episodes // 2 :] = 1

    obs, _ = wrapper.reset(options={"agent_side": agent_sides})

    finished_rewards = torch.zeros(n_episodes, device=device)
    active_mask = torch.ones(n_episodes, dtype=torch.bool, device=device)

    while active_mask.any():
        with torch.no_grad():
            actions = agent_policy.act(obs, deterministic=False)

        obs, rewards, terminated, _, _ = wrapper.step(actions)

        just_finished = terminated & active_mask
        if just_finished.any():
            finished_rewards[just_finished] = rewards[just_finished]

        active_mask = active_mask & ~terminated

    wins = (finished_rewards == 1.0).sum().item()
    losses = (finished_rewards == -1.0).sum().item()
    draws = (finished_rewards == 0.0).sum().item()

    return {
        "validation/vs_benchmark/win_rate": wins / n_episodes,
        "validation/vs_benchmark/loss_rate": losses / n_episodes,
        "validation/vs_benchmark/draw_rate": draws / n_episodes,
        "validation/vs_benchmark/score_rate": (wins + 0.5 * draws) / n_episodes,
        "validation/vs_benchmark/games_played": n_episodes,
    }
