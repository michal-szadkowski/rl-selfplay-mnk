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
from model_export import ModelExporter


def train_mnk():
    default_config = {
        "mnk": (9, 9, 5),
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "n_steps": 512,
        "training_iterations": 4000,
        "validation_interval": 50,
        "validation_episodes": 100,
        "benchmark_update_threshold": 0.65,
        "opponent_pool_size": 15,
        "num_envs": 12,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with wandb.init(
        config=default_config, project="mnk_vector_a2c", group="vec", dir="./wnb"
    ) as run:
        # Initialize model exporter
        model_exporter = ModelExporter(run.name or None)

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
        run.watch(agent.network)

        network_creator = lambda: ActorCriticModule(obs_shape, action_dim)

        opponent_pool = OpponentPool(
            network_creator, max_size=run.config.opponent_pool_size, device=device
        )

        benchmark_policy = NNPolicy(deepcopy(agent.network))

        opponent_pool.add_opponent(agent.network)

        for i in range(run.config.training_iterations):
            try:
                # Sample opponent
                opponent, opponent_idx = opponent_pool.sample_opponent()
                train_env.unwrapped.set_opponent(opponent)

                # Train and get metrics
                metrics = agent.learn(train_env)

                # Update opponent statistics
                # Agent's reward is opponent's negative reward (zero-sum game)
                opponent_reward = -metrics.mean_reward
                opponent_pool.update_opponent_stats(opponent_idx, opponent_reward)
                log_pool_stats(run, opponent_pool, i)

                if should_add_to_pool(metrics, i):
                    opponent_pool.add_opponent(agent.network)

                log_training_metrics(run, metrics, i)

                # Validate agent performance periodically
                if i > 0 and i % run.config.validation_interval == 0:
                    print(f"--- Running validation at step {i} ---")
                    validation_res = run_validation(
                        mnk,
                        run.config.validation_episodes,
                        agent,
                        benchmark_policy,
                        device,
                        seed=i,
                    )
                    run.log(validation_res, step=i)

                    if (
                        validation_res["validation/vs_benchmark/win_rate"]
                        > run.config.benchmark_update_threshold
                    ):
                        print(f"--- New benchmark agent at step {i}! ---")
                        benchmark_policy = NNPolicy(deepcopy(agent.network))

                        # Export model that broke benchmark
                        model_exporter.export_model(
                            agent.network, i, is_benchmark_breaker=True
                        )

                        run.log({"validation/new_benchmark_step": i}, step=i)

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


def should_add_to_pool(metrics, iteration):
    """Simple but effective opponent addition."""
    # Add good performing agents
    if metrics.mean_reward > 0.1:
        return True

    # Add periodically for diversity (but less frequently)
    if iteration % 15 == 0:
        return True

    return False


def log_training_metrics(run, metrics: TrainingMetrics, iteration):
    """Log training metrics."""
    print(
        f"Iteration {iteration}: Mean reward = {metrics.mean_reward:.3f}, Mean length = {metrics.mean_length:.1f}"
    )
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
