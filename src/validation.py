import numpy as np
from pettingzoo.utils.env import AECEnv

from selfplay.self_play_wrapper import Policy


def play_episode(env: AECEnv, agent_policy: Policy, opponent_policy: Policy):

    # Reset środowiska i losowe przypisanie graczy
    original_agents = env.possible_agents
    env.reset()
    agent_player_id = np.random.choice([0, 1])

    policies = {
        original_agents[agent_player_id]: agent_policy,
        original_agents[1 - agent_player_id]: opponent_policy,
    }

    # Rozegraj jedną partię
    for agent_name in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            # Gra się skończyła, przejdź do następnej
            break

        # Wybierz akcję zgodnie z polityką dla danego agenta
        action = policies[agent_name].act(obs)
        env.step(action)

    # Po zakończeniu partii, zbierz wyniki
    agent_final_reward = env._cumulative_rewards[original_agents[agent_player_id]]
    return agent_final_reward


def validate_episodes(
    env: AECEnv,
    agent_policy: Policy,
    opponent_policy: Policy,
    num_episodes: int,
):
    """
    Rozgrywa num_episodes gier pomiędzy agent_policy a opponent_policy,
    zwracając statystyki (wygrane, przegrane, remisy).
    """
    stats = {"wins": 0, "losses": 0, "draws": 0}

    for _ in range(num_episodes):
        agent_final_reward = play_episode(env, agent_policy, opponent_policy)

        if agent_final_reward > 0:
            stats["wins"] += 1
        elif agent_final_reward < 0:
            stats["losses"] += 1
        else:
            stats["draws"] += 1

    # Oblicz wskaźniki procentowe
    if num_episodes > 0:
        stats["win_rate"] = stats["wins"] / num_episodes
        stats["loss_rate"] = stats["losses"] / num_episodes
        stats["draw_rate"] = stats["draws"] / num_episodes
    else:
        stats["win_rate"] = 0
        stats["loss_rate"] = 0
        stats["draw_rate"] = 0

    return stats
