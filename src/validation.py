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
        seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Plays a single episode between agent_policy and opponent_policy.
    Returns a dictionary with results: 'agent_reward', 'opponent_reward', 'outcome'
    where outcome is 'win', 'loss', 'draw' from agent's perspective.
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

    # Track cumulative rewards for each agent
    cumulative_rewards = {agent: 0.0 for agent in agents}

    # Play the episode
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        # Accumulate reward for the current agent
        cumulative_rewards[agent] += reward

        if not (termination or truncation):
            # Get action from the policy and step
            action = policies[agent].act(obs)
            env.step(action)
        else:
            env.step(None)

    # Determine final rewards
    agent_reward = cumulative_rewards[agents[0]] if agent_is_first else cumulative_rewards[agents[1]]
    opponent_reward = cumulative_rewards[agents[1]] if agent_is_first else cumulative_rewards[agents[0]]

    # Determine outcome from agent's perspective
    if agent_reward > opponent_reward:
        outcome = 'win'
    elif agent_reward < opponent_reward:
        outcome = 'loss'
    else:
        outcome = 'draw'

    return {
        'agent_reward': agent_reward,
        'opponent_reward': opponent_reward,
        'outcome': outcome,
    }


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
            env, agent_policy, opponent_policy,
            agent_is_first=True,
            seed=episode_seed
        )
        results.append(result)

    # Second half: agent plays as second player
    for episode in range(episodes_as_second):
        episode_seed = seed + episodes_as_first + episode if seed is not None else None
        result = play_single_episode(
            env, agent_policy, opponent_policy,
            agent_is_first=False,
            seed=episode_seed
        )
        results.append(result)

    # Aggregate statistics
    wins = sum(1 for r in results if r['outcome'] == 'win')
    losses = sum(1 for r in results if r['outcome'] == 'loss')
    draws = sum(1 for r in results if r['outcome'] == 'draw')

    win_rate = wins / num_episodes
    loss_rate = losses / num_episodes
    draw_rate = draws / num_episodes

    avg_agent_reward = np.mean([r['agent_reward'] for r in results])
    avg_opponent_reward = np.mean([r['opponent_reward'] for r in results])

    return {
        'num_episodes': num_episodes,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'draw_rate': draw_rate,
        'avg_agent_reward': avg_agent_reward,
        'avg_opponent_reward': avg_opponent_reward,
    }


def run_validation(mnk, n_episodes, agent, benchmark_policy, device, seed=None):
    """Run validation for the current agent against random and benchmark opponents."""
    # Create validation environment
    validation_env = create_mnk_env(m=mnk[0], n=mnk[1], k=mnk[2])

    # Set agent to eval mode
    agent.network.eval()

    # Create policies for validation
    current_policy = NNPolicy(deepcopy(agent.network), device=device)
    random_policy = RandomPolicy(validation_env.action_space(validation_env.possible_agents[0]))

    # Set benchmark policy to eval mode
    if hasattr(benchmark_policy, 'model'):
        benchmark_policy.model.eval()

    # 1. Validate against a random opponent
    random_stats = validate_agent(
        validation_env,
        current_policy,
        random_policy,
        n_episodes,
        seed=seed
    )

    # 2. Validate against the benchmark agent
    benchmark_stats = validate_agent(
        validation_env,
        current_policy,
        benchmark_policy,
        n_episodes,
        seed=seed + 1000 if seed is not None else None
    )

    # Print validation results in a table format
    print("\nValidation Results:")
    print("-" * 60)
    print(f"{'Opponent':<12} | {'Win Rate':<9} | {'Loss Rate':<10} | {'Draw Rate':<10} | {'Avg Reward':<11}")
    print("-" * 60)
    print(
        f"{'Random':<12} | {random_stats['win_rate']:<9.3f} | {random_stats['loss_rate']:<10.3f} | {random_stats['draw_rate']:<10.3f} | {random_stats['avg_agent_reward']:<11.3f}")
    print(
        f"{'Benchmark':<12} | {benchmark_stats['win_rate']:<9.3f} | {benchmark_stats['loss_rate']:<10.3f} | {benchmark_stats['draw_rate']:<10.3f} | {benchmark_stats['avg_agent_reward']:<11.3f}")
    print("-" * 60)

    # Set agent back to train mode
    agent.network.train()

    return {
        "validation/vs_random/win_rate": random_stats["win_rate"],
        "validation/vs_random/loss_rate": random_stats["loss_rate"],
        "validation/vs_random/draw_rate": random_stats["draw_rate"],
        "validation/vs_random/avg_reward": random_stats["avg_agent_reward"],
        "validation/vs_benchmark/win_rate": benchmark_stats["win_rate"],
        "validation/vs_benchmark/loss_rate": benchmark_stats["loss_rate"],
        "validation/vs_benchmark/draw_rate": benchmark_stats["draw_rate"],
        "validation/vs_benchmark/avg_reward": benchmark_stats["avg_agent_reward"],
    }
