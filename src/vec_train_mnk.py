from copy import deepcopy

import torch
import wandb
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from src.alg.ppo import PPOAgent, TrainingMetrics
from src.alg.resnet import SimpleResNetActorCritic
from src.env.mnk_game_env import create_mnk_env
from src.selfplay.opponent_pool import OpponentPool
from src.selfplay.policy import NNPolicy, BatchNNPolicy
from src.selfplay.vector_mnk_self_play import VectorMnkSelfPlayWrapper
from src.validation import run_validation
from src.model_export import ModelExporter


def train_mnk():
    default_config = {
        "mnk": (9, 9, 5),
        "learning_rate": 1e-4,
        "gamma": 0.98,
        "batch_size": 1024,
        "n_steps": 512,
        "ppo_epochs": 4,
        "total_environment_steps": 50_000_000,
        "validation_interval": 25,
        "validation_episodes": 100,
        "benchmark_update_threshold": 0.60,
        "num_envs": 16,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with wandb.init(config=default_config, project="mnk_995") as run:
        # Initialize model exporter
        model_exporter = ModelExporter(run.name or None)

        mnk = run.config.mnk

        def env_fn():
            return create_mnk_env(m=mnk[0], n=mnk[1], k=mnk[2])

        # Initialize vectorized self-play environment
        train_env = VectorMnkSelfPlayWrapper(
            m=mnk[0], n=mnk[1], k=mnk[2], n_envs=run.config.num_envs
        )
        train_env = RecordEpisodeStatistics(train_env)

        obs_shape = train_env.single_observation_space["observation"].shape
        action_dim = train_env.single_action_space.n

        # Initialize A2C agent with neural network
        network = SimpleResNetActorCritic(obs_shape, action_dim)
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
            ppo_epochs=run.config.ppo_epochs,
        )
        run.watch(agent.network)

        benchmark_policy = NNPolicy(deepcopy(agent.network))

        # Initialize opponent pool
        opponent_pool = OpponentPool(max_size=5)
        opponent_pool.add_opponent(BatchNNPolicy(deepcopy(agent.network)))

        steps_per_iteration = run.config.num_envs * run.config.n_steps
        total_iterations = run.config.total_environment_steps // steps_per_iteration
        print(f"Starting training for {total_iterations}")

        for i in range(total_iterations):
            current_env_steps = (i + 1) * steps_per_iteration
            try:
                # Select random opponent from pool
                opponent = opponent_pool.get_random_opponent()
                train_env.unwrapped.set_opponent(opponent)

                # Train and get metrics
                metrics = agent.learn(train_env)

                log_training_metrics(run, metrics, i, current_env_steps)

                if i % 10 == 0 or metrics.mean_reward > 0.3:
                    opponent_pool.add_opponent(BatchNNPolicy(deepcopy(agent.network)))

                # Validate agent performance periodically
                if i > 0 and i % run.config.validation_interval == 0:
                    print(
                        f"--- Running validation at step {i} ({current_env_steps:,} env steps) ---"
                    )
                    validation_res = run_validation(
                        mnk,
                        run.config.validation_episodes,
                        agent,
                        benchmark_policy,
                        device,
                        seed=i,
                    )
                    run.log(validation_res, step=current_env_steps)

                    if (
                        validation_res["validation/vs_benchmark/win_rate"]
                        > run.config.benchmark_update_threshold
                    ):
                        print(
                            f"--- New benchmark agent at step {i} ({current_env_steps:,} env steps)! ---"
                        )
                        benchmark_policy = NNPolicy(deepcopy(agent.network))

                        # Export model that broke benchmark
                        model_exporter.export_model(
                            agent.network, i, is_benchmark_breaker=True
                        )

                        run.log({"validation/new_benchmark_step": i}, step=current_env_steps)

            except Exception as e:
                handle_training_error(run, e, i, current_env_steps)
                continue


def log_training_metrics(run, metrics: TrainingMetrics, iteration, env_steps):
    """Log training metrics."""
    print(
        f"Iter {iteration} | {env_steps:,} steps | "
        f"reward: {metrics.mean_reward:.3f} | "
        f"length: {metrics.mean_length:.1f} | "
        f"entropy: {metrics.entropy_loss:.4f} | "
        f"grad_norm: {metrics.grad_norm:.3f} | "
        f"clip: {metrics.clip_fraction:.3f}"
    )

    run.log(
        {
            "training/mean_reward": metrics.mean_reward,
            "training/mean_length": metrics.mean_length,
            "training/actor_loss": metrics.actor_loss,
            "training/critic_loss": metrics.critic_loss,
            "training/entropy_loss": metrics.entropy_loss,
            "training/grad_norm": metrics.grad_norm,
            "training/clip_fraction": metrics.clip_fraction,
        },
        step=env_steps,
    )


def handle_training_error(run, error, iteration, env_steps):
    """Handle training exceptions and log to W&B."""
    error_msg = f"Error in iteration {iteration}: {str(error)}"
    print(error_msg)
    import traceback

    traceback.print_exc()
    run.log(
        {
            "error/iteration": iteration,
            "error/message": str(error),
            "error/traceback": traceback.format_exc(),
        },
        step=env_steps,
    )


if __name__ == "__main__":
    train_mnk()
