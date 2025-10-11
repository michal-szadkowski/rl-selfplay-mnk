import os
from copy import deepcopy

import torch
import wandb
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from alg.ppo import PPOAgent, ActorCriticModule, TrainingMetrics
from env.mnk_game_env import create_mnk_env
from selfplay.policy import NNPolicy
from selfplay.vector_self_play_wrapper import VectorSelfPlayWrapper
from selfplay.opponent_pool import OpponentPool
from validation import run_validation


def train_mnk():
    default_config = {
        "mnk": (9, 9, 5),
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "n_steps": 512,
        "training_iterations": 750,
        "validation_interval": 10,
        "validation_episodes": 100,
        "benchmark_update_threshold": 0.65,
        "opponent_pool_size": 5,
        "num_envs": 12,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with wandb.init(
        config=default_config, project="mnk_vector_a2c", group="vec", dir="./wnb"
    ) as run:
        mnk = run.config.mnk

        def env_fn():
            return create_mnk_env(m=mnk[0], n=mnk[1], k=mnk[2])

        # Initialize vectorized self-play environment
        train_env = VectorSelfPlayWrapper(env_fn, n_envs=run.config.num_envs)
        train_env = RecordEpisodeStatistics(train_env)

        obs_shape = train_env.single_observation_space["observation"].shape
        action_dim = train_env.single_action_space.n

        # Initialize A2C agent with neural network
        network = ActorCriticModule(obs_shape, action_dim)
        agent = PPOAgent(
            obs_shape,
            action_dim,
            network,
            n_steps=run.config.n_steps,
            learning_rate=run.config.learning_rate,
            gamma=run.config.gamma,
            batch_size=run.config.batch_size,
            device=device,
            num_envs=run.config.num_envs,
        )

        network_creator = lambda: ActorCriticModule(obs_shape, action_dim)

        opponent_pool = OpponentPool(
            network_creator, max_size=run.config.opponent_pool_size, device=device
        )

        run.watch(agent.network)

        benchmark_policy = NNPolicy(deepcopy(agent.network))

        for i in range(run.config.training_iterations):
            try:
                # Add agent to pool periodically
                if i % 3 == 0:
                    opponent_pool.add_opponent(agent.network)

                # Sample opponent
                opponent, opponent_idx = opponent_pool.sample_opponent()
                train_env.unwrapped.set_opponent(opponent)

                # Train and get metrics
                metrics = agent.learn(train_env)

                # Update opponent statistics
                # Agent's reward is opponent's negative reward (zero-sum game)
                opponent_reward = -metrics.mean_reward
                opponent_pool.update_opponent_stats(opponent_idx, opponent_reward)

                print(
                    f"Iteration {i}: Mean reward = {metrics.mean_reward:.3f}, Mean length = {metrics.mean_length:.1f}"
                )

                # Log pool statistics periodically
                if i % 10 == 0:
                    log_pool_stats(run, opponent_pool, i)

                # Log training metrics
                log_training_metrics(run, metrics, i)

                # Validate agent performance periodically
                if i > 0 and i % run.config.validation_interval == 0:
                    benchmark_policy = run_validation_and_update_benchmark(
                        run,
                        mnk,
                        run.config.validation_episodes,
                        agent,
                        benchmark_policy,
                        device,
                        i,
                        opponent_pool,
                        run.config.benchmark_update_threshold,
                    )

            except Exception as e:
                error_msg = f"Error in iteration {i}: {str(e)}"
                print(error_msg)
                import traceback

                traceback.print_exc()
                run.log(
                    {
                        "error/iteration": i,
                        "error/message": str(e),
                        "error/traceback": traceback.format_exc(),
                    },
                    step=i,
                )
                # Continue to next iteration or break if critical
                continue


def run_validation_and_update_benchmark(
    run,
    mnk,
    validation_episodes,
    agent,
    benchmark_policy,
    device,
    iteration,
    opponent_pool,
    benchmark_update_threshold,
):
    print(f"--- Running validation at step {iteration} ---")

    validation_res = run_validation(
        mnk, validation_episodes, agent, benchmark_policy, device, seed=iteration
    )
    run.log(validation_res, step=iteration)

    if validation_res["validation/vs_benchmark/win_rate"] > benchmark_update_threshold:
        print(f"--- New benchmark agent at step {iteration}! ---")
        benchmark_policy = NNPolicy(deepcopy(agent.network))
        opponent_pool.add_opponent(agent.network)
        save_benchmark_model(agent, run.name or "", iteration)
        run.log({"validation/new_benchmark_step": iteration}, step=iteration)

    return benchmark_policy


def save_benchmark_model(agent, name, step):
    """Saves the agent's network as a new benchmark model."""
    dirname, _ = os.path.split(os.path.abspath(__file__))
    model_dir = os.path.join(dirname, "models", name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"benchmark_step_{step}.pt")
    torch.save(agent.network.state_dict(), model_path)
    print(f"Saved new benchmark model to {model_path}")


def log_training_metrics(run, metrics: TrainingMetrics, iteration):
    """Log training metrics."""
    run.log(
        {
            "training/mean_reward": metrics.mean_reward,
            "training/mean_length": metrics.mean_length,
            "training/actor_loss": metrics.actor_loss,
            "training/critic_loss": metrics.critic_loss,
            "training/entropy_loss": metrics.entropy_loss,
        },
        step=iteration,
    )


def log_pool_stats(run, opponent_pool, iteration):
    """Log pool statistics."""
    pool_stats = opponent_pool.get_pool_stats()
    run.log(
        {
            "pool/size": pool_stats["size"],
            "pool/avg_mean_reward": pool_stats["avg_mean_reward"],
            "pool/avg_games_played": pool_stats["avg_games_played"],
            "pool/total_games": pool_stats["total_games"],
        },
        step=iteration,
    )


if __name__ == "__main__":
    train_mnk()
