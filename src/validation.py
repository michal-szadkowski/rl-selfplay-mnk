from copy import deepcopy
from typing import Dict, Any, Optional

import numpy as np
from pettingzoo.utils.env import AECEnv

from env.mnk_game_env import create_mnk_env
from selfplay.policy import NNPolicy, RandomPolicy, Policy


def play_single_episode(
    env: AECEnv,
    agent_policy: Policy,
    opponent_policy: Policy,
    agent_is_first: Optional[bool] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Plays a single episode between agent_policy and opponent_policy.
    Returns the outcome: 'win', 'loss', 'draw' from agent's perspective.
    """
    # Reset the environment
    env.reset(seed=seed)

    # Randomly assign which policy is player 1 or 2
    agents = env.possible_agents
    if len(agents) != 2:
        raise ValueError("Validation currently supports only 2-player games.")

    # Determine which policy plays first
    if agent_is_first is None:
        rng = np.random.default_rng(seed)
        agent_is_first = rng.choice([True, False])

    policies = {
        agents[0]: agent_policy if agent_is_first else opponent_policy,
        agents[1]: opponent_policy if agent_is_first else agent_policy,
    }

    # Track final rewards to determine outcome
    final_rewards = {}

    # Play the episode
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        final_rewards[agent] = reward

        if not (termination or truncation):
            # Get action from the policy and step
            action = policies[agent].act(obs)
            env.step(action)
        else:
            env.step(None)

    # Determine outcome from agent's perspective
    agent_final_reward = (
        final_rewards[agents[0]] if agent_is_first else final_rewards[agents[1]]
    )

    if agent_final_reward > 0:
        outcome = "win"
    elif agent_final_reward < 0:
        outcome = "loss"
    else:
        outcome = "draw"

    return {"outcome": outcome}


def validate_agent(
    env: AECEnv,
    agent_policy: Policy,
    opponent_policy: Policy,
    num_episodes: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Validates the agent_policy against opponent_policy over num_episodes.
    Returns statistics: wins, losses, draws, win_rate, etc.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    results = []
    episodes_as_first = num_episodes // 2
    episodes_as_second = num_episodes - episodes_as_first

    # First half: agent plays as first player
    for episode in range(episodes_as_first):
        episode_seed = seed + episode if seed is not None else None
        result = play_single_episode(
            env, agent_policy, opponent_policy, agent_is_first=True, seed=episode_seed
        )
        results.append(result)

    # Second half: agent plays as second player
    for episode in range(episodes_as_second):
        episode_seed = seed + episodes_as_first + episode if seed is not None else None
        result = play_single_episode(
            env, agent_policy, opponent_policy, agent_is_first=False, seed=episode_seed
        )
        results.append(result)

    # Aggregate statistics
    wins = sum(1 for r in results if r["outcome"] == "win")
    losses = sum(1 for r in results if r["outcome"] == "loss")
    draws = sum(1 for r in results if r["outcome"] == "draw")

    win_rate = wins / num_episodes
    loss_rate = losses / num_episodes
    draw_rate = draws / num_episodes

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
    }


def run_validation(mnk, n_episodes, agent, benchmark_policy, device, seed=None):
    """Run validation for the current agent against random and benchmark opponents."""
    # Create validation environment
    validation_env = create_mnk_env(m=mnk[0], n=mnk[1], k=mnk[2])

    # Set agent to eval mode
    agent.network.eval()

    # Create policies for validation
    current_policy = NNPolicy(deepcopy(agent.network), device=device)
    random_policy = RandomPolicy(
        validation_env.action_space(validation_env.possible_agents[0])
    )

    # Set benchmark policy to eval mode
    if hasattr(benchmark_policy, "model"):
        benchmark_policy.model.eval()

    # 1. Validate against a random opponent
    random_stats = validate_agent(
        validation_env, current_policy, random_policy, n_episodes, seed=seed
    )

    # 2. Validate against the benchmark agent
    benchmark_stats = validate_agent(
        validation_env,
        current_policy,
        benchmark_policy,
        n_episodes,
        seed=seed + 1000 if seed is not None else None,
    )

    # Calculate score rates (win * 1.0 + draw * 0.5)
    random_score_rate = (random_stats["wins"] * 1.0 + random_stats["draws"] * 0.5) / n_episodes
    benchmark_score_rate = (
        benchmark_stats["wins"] * 1.0 + benchmark_stats["draws"] * 0.5
    ) / n_episodes

    # Print validation results in a table format
    print("\nValidation Results:")
    print("-" * 55)
    print(f"{'Opponent':<12} | {'Win Rate':<9} | {'Draw Rate':<10} | {'Score Rate':<11}")
    print("-" * 55)
    print(
        f"{'Random':<12} | {random_stats['win_rate']:<9.3f} | {random_stats['draw_rate']:<10.3f} | {random_score_rate:<11.3f}"
    )
    print(
        f"{'Benchmark':<12} | {benchmark_stats['win_rate']:<9.3f} | {benchmark_stats['draw_rate']:<10.3f} | {benchmark_score_rate:<11.3f}"
    )
    print("-" * 55)

    # Set agent back to train mode
    agent.network.train()

    return {
        "validation/vs_random/win_rate": random_stats["win_rate"],
        "validation/vs_random/draw_rate": random_stats["draw_rate"],
        "validation/vs_random/win_count": random_stats["wins"],
        "validation/vs_random/draw_count": random_stats["draws"],
        "validation/vs_random/score_rate": random_score_rate,
        "validation/vs_benchmark/win_rate": benchmark_stats["win_rate"],
        "validation/vs_benchmark/draw_rate": benchmark_stats["draw_rate"],
        "validation/vs_benchmark/win_count": benchmark_stats["wins"],
        "validation/vs_benchmark/draw_count": benchmark_stats["draws"],
        "validation/vs_benchmark/score_rate": benchmark_score_rate,
    }
