from copy import deepcopy

import torch
import wandb
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from alg.ppo import PPOAgent, TrainingMetrics
from alg.entropy_scheduler import EntropyScheduler
from alg.lr_scheduler import create_lr_scheduler
from selfplay.opponent_pool import OpponentPool
from selfplay.policy import NNPolicy, BatchNNPolicy
from selfplay.vector_mnk_self_play import VectorMnkSelfPlayWrapper
from validation import run_validation
from model_export import ModelExporter, create_model_from_architecture


def setup_environment(config):
    """Initialize vectorized self-play environment and return env, obs_shape, action_dim."""
    train_env = VectorMnkSelfPlayWrapper(
        m=config.mnk[0], n=config.mnk[1], k=config.mnk[2], n_envs=config.num_envs
    )
    train_env = RecordEpisodeStatistics(train_env)

    obs_shape = train_env.single_observation_space["observation"].shape
    action_dim = train_env.single_action_space.n

    return train_env, obs_shape, action_dim


def create_agent(config, obs_shape, action_dim, device):
    """Initialize PPO agent with network, optimizer, and scheduler."""
    network = create_model_from_architecture(
        config.architecture_name, obs_shape=obs_shape, action_dim=action_dim
    )

    optimizer = torch.optim.AdamW(
        network.parameters(), lr=config.learning_rate, weight_decay=1e-4
    )

    lr_scheduler = create_lr_scheduler(
        optimizer, config.lr_warmup_steps, config.num_envs, config.n_steps
    )

    entropy_scheduler = EntropyScheduler(
        initial_coef=config.entropy_coef,
        schedule=config.entropy_coef_schedule,
        num_envs=config.num_envs,
        n_steps=config.n_steps,
    )

    agent = PPOAgent(
        obs_shape,
        action_dim,
        network,
        n_steps=config.n_steps,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        batch_size=config.batch_size,
        device=device,
        num_envs=config.num_envs,
        ppo_epochs=config.ppo_epochs,
        entropy_coef=config.entropy_coef,
        lr_scheduler=lr_scheduler,
        entropy_scheduler=entropy_scheduler,
        optimizer=optimizer,
        clip_range=config.clip_range,
    )

    return agent


def train_mnk():
    default_config = {
        "mnk": (9, 9, 5),
        # lr
        "learning_rate": 1e-4,
        "lr_warmup_steps": 500_000,
        # entropy
        "entropy_coef": 0.01,
        "entropy_coef_schedule": {
            "type": "linear",
            "params": {"final_coef": 0.001, "total_steps": 40_000_000},
        },
        # ppo
        "gamma": 0.98,
        "clip_range": 0.2,
        "batch_size": 1024,
        "n_steps": 256,
        "ppo_epochs": 4,
        "total_environment_steps": 50_000_000,
        "num_envs": 32,
        # validation
        "benchmark_update_threshold_score": 0.65,
        "validation_interval": 25,
        "validation_episodes": 100,
        # selfplay
        "opponent_pool": 5,
        #
        "architecture_name": "transformer_s",
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with wandb.init(config=default_config, project="mnk_995") as run:
        model_exporter = ModelExporter(run.name or None)

        train_env, obs_shape, action_dim = setup_environment(run.config)

        agent = create_agent(run.config, obs_shape, action_dim, device)
        run.watch(agent.network)

        benchmark_policy = NNPolicy(deepcopy(agent.network))

        opponent_pool = OpponentPool(max_size=run.config.opponent_pool)
        opponent_pool.add_opponent(BatchNNPolicy(deepcopy(agent.network)))

        steps_per_iteration = run.config.num_envs * run.config.n_steps
        total_iterations = run.config.total_environment_steps // steps_per_iteration
        print(f"Starting training for {total_iterations}")

        current_env_steps = 0
        for i in range(total_iterations):
            try:

                # Select random opponent from pool
                opponent = opponent_pool.get_random_opponent()
                train_env.unwrapped.set_opponent(opponent)

                # Train and get metrics
                metrics = agent.learn(train_env)

                log_training_metrics(
                    run, metrics, i, current_env_steps, agent.entropy_coef, agent
                )

                if i % 10 == 0 or metrics.mean_reward > 0.3:
                    opponent_pool.add_opponent(BatchNNPolicy(deepcopy(agent.network)))

                # Validate agent performance periodically
                if i > 0 and i % run.config.validation_interval == 0:
                    print(
                        f"--- Running validation at step {i} ({current_env_steps:,} env steps) ---"
                    )
                    validation_res = run_validation(
                        run.config.mnk,
                        run.config.validation_episodes,
                        agent,
                        benchmark_policy,
                        device,
                        seed=i,
                    )
                    run.log(validation_res, step=current_env_steps)

                    score_rate = validation_res["validation/vs_benchmark/score_rate"]

                    if score_rate > run.config.benchmark_update_threshold_score:
                        print(
                            f"--- New benchmark agent at step {i} (Score Rate: {score_rate:.2f})! ---"
                        )
                        benchmark_policy = NNPolicy(deepcopy(agent.network))

                        # Export model that broke benchmark
                        model_exporter.export_model(
                            agent.network, i, is_benchmark_breaker=True
                        )

                        run.log({"validation/new_benchmark_step": i}, step=current_env_steps)
                    else:
                        model_exporter.export_model(
                            agent.network, i, is_benchmark_breaker=False
                        )

            except Exception as e:
                handle_training_error(run, e, i, current_env_steps)
                continue

            current_env_steps = (i + 1) * steps_per_iteration


def log_training_metrics(
    run,
    metrics: TrainingMetrics,
    iteration,
    env_steps,
    entropy_coef,
    agent,
):
    """Log training metrics."""
    current_lr = agent.optimizer.param_groups[0]["lr"]

    print(
        f"Iter {iteration} | {env_steps:,} steps | "
        f"reward: {metrics.mean_reward:.3f} | "
        f"length: {metrics.mean_length:.1f} | "
        f"entropy: {metrics.entropy_loss:.4f} | "
        f"entropy_coef: {entropy_coef:.4f} | "
        f"lr: {current_lr:.6f} | "
        f"grad_norm: {metrics.grad_norm:.3f} | "
        f"clip: {metrics.clip_fraction:.3f} | "
        f"explained_var: {metrics.explained_variance:.3f} | "
        f"approx_kl: {metrics.approx_kl:.4f}"
    )

    run.log(
        {
            "training/mean_reward": metrics.mean_reward,
            "training/mean_length": metrics.mean_length,
            "training/actor_loss": metrics.actor_loss,
            "training/critic_loss": metrics.critic_loss,
            "training/entropy_loss": metrics.entropy_loss,
            "training/entropy_coef": entropy_coef,
            "training/learning_rate": current_lr,
            "training/grad_norm": metrics.grad_norm,
            "training/clip_fraction": metrics.clip_fraction,
            "training/explained_variance": metrics.explained_variance,
            "training/approx_kl": metrics.approx_kl,
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
