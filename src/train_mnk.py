import torch
from copy import deepcopy
from gymnasium.wrappers.vector import RecordEpisodeStatistics
import wandb
import os
import gymnasium as gym
from env.mnk_game_env import create_mnk_env
from selfplay.self_play_wrapper import SelfPlayWrapper, NNPolicy, RandomPolicy
from alg.ppo import PPOAgent, ActorCriticModule
from validation import validate_episodes


def create_wrapped_env(mnk, render_mode=None):
    """Creates and wraps the MNK environment for training."""
    env = create_mnk_env(m=mnk[0], n=mnk[1], k=mnk[2], render_mode=render_mode)
    env = SelfPlayWrapper(env)
    return env


def train_mnk():
    # Define hyperparameters
    config = {
        "mnk": (9, 9, 5),
        "learning_rate": 8e-5,
        "gamma": 0.99,
        "batch_size": 64,
        "n_steps": 1024,
        "training_iterations": 1000,
        "validation_interval": 10,
        "validation_episodes": 100,
        "benchmark_update_threshold": 0.55,
        "opponent_pool_size": 10,
        "num_envs": 8,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with wandb.init(project="mnk_a2c", config=config, group="vec") as run:
        mnk = run.config.mnk

        def make_env():
            env = create_wrapped_env(mnk)
            return env

        env_fns = [make_env for _ in range(run.config.num_envs)]
        train_env = gym.vector.SyncVectorEnv([make_env for _ in range(run.config.num_envs)])
        train_env = RecordEpisodeStatistics(train_env)

        # Instantiate the A2CAgent
        obs_shape = train_env.single_observation_space["observation"].shape
        action_dim = train_env.single_action_space.n
        network = ActorCriticModule(obs_shape, action_dim)
        agent = PPOAgent(obs_shape, action_dim, network, n_steps=run.config.n_steps, learning_rate=run.config.learning_rate,
                         gamma=run.config.gamma, batch_size=run.config.batch_size, device=device, num_envs=run.config.num_envs)

        run.watch(agent.network)

        # Create benchmark agent and a reusable random policy for validation
        benchmark_policy = NNPolicy(deepcopy(agent.network))

        # --- training loop ---
        for i in range(run.config.training_iterations):
            mean_reward, mean_length, actor_loss, critic_loss, entropy_loss = agent.learn(train_env)

            print(f"Iteration {i + 1}: Mean reward = {mean_reward}, Mean length = {mean_length}")

            run.log(
                {
                    "training/mean_reward": mean_reward,
                    "training/mean_length": mean_length,
                    "training/actor_loss": actor_loss,
                    "training/critic_loss": critic_loss,
                    "training/entropy_loss": entropy_loss,
                },
                step=i,
            )

            if i > 0 and i % run.config.validation_interval == 0:
                print(f"--- Running validation at step {i} ---")

                validation_res = validate(mnk, run.config.validation_episodes, agent, benchmark_policy, device)
                run.log(validation_res, step=i)

                if validation_res["validation/vs_benchmark/win_rate"] > run.config.benchmark_update_threshold:
                    print(f"--- New benchmark agent at step {i}! ---")
                    benchmark_policy = NNPolicy(deepcopy(agent.network))

                    save_benchmark_model(agent, run.name, i)

                    # Update opponent pool in each parallel environment
                    for env in train_env.env.envs:
                        env.opponent_pool.append(benchmark_policy)
                        if len(env.opponent_pool) > run.config.opponent_pool_size:
                            env.opponent_pool.pop(0)

                    run.log({"validation/new_benchmark_step": i}, step=i)


def save_benchmark_model(agent, name, step):
    """Saves the agent's network as a new benchmark model."""
    dirname, _ = os.path.split(os.path.abspath(__file__))
    model_dir = os.path.join(dirname, "models", name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"benchmark_step_{step}.pt")
    torch.save(agent.network.state_dict(), model_path)
    print(f"Saved new benchmark model to {model_path}")


def validate(mnk, n_episodes, agent, benchmark_policy, device):
    validation_env = create_mnk_env(m=mnk[0], n=mnk[1], k=mnk[2])

    agent.network.eval()

    current_policy = NNPolicy(agent.network, device=device)
    random_policy = RandomPolicy(validation_env.action_space(validation_env.possible_agents[0]))

    # 1. Validate against a random opponent
    random_stats = validate_episodes(
        validation_env,
        current_policy,
        random_policy,
        n_episodes,
    )
    print(f"Win rate vs random = {random_stats['win_rate']:.2f}/{random_stats['draw_rate']:.2f}")

    # 2. Validate against the benchmark agent
    benchmark_stats = validate_episodes(
        validation_env,
        current_policy,
        benchmark_policy,
        n_episodes,
    )
    print(f"Win rate vs benchmark = {benchmark_stats['win_rate']:.2f}/{benchmark_stats['draw_rate']:.2f}")

    agent.network.train()  # Set model back to training mode
    return (
        {
            "validation/vs_random/win_rate": random_stats["win_rate"],
            "validation/vs_random/loss_rate": random_stats["loss_rate"],
            "validation/vs_random/draw_rate": random_stats["draw_rate"],
            "validation/vs_benchmark/win_rate": benchmark_stats["win_rate"],
            "validation/vs_benchmark/loss_rate": benchmark_stats["loss_rate"],
            "validation/vs_benchmark/draw_rate": benchmark_stats["draw_rate"],
        }
    )


if __name__ == "__main__":
    train_mnk()
