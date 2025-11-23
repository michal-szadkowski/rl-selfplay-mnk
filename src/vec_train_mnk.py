from copy import deepcopy
import random

import torch
import wandb

from alg.ppo import PPOAgent, TrainingMetrics
from alg.entropy_scheduler import EntropyScheduler
from alg.lr_scheduler import create_lr_scheduler
from selfplay.opponent_pool import OpponentPool
from selfplay.policy import NNPolicy
from selfplay.torch_self_play_wrapper import TorchSelfPlayWrapper
from env.torch_vector_mnk_env import TorchVectorMnkEnv
from selfplay.validation import validate_gpu
from model_export import ModelExporter, create_model_from_architecture


def setup_environment(config):
    base_env = TorchVectorMnkEnv(
        m=config.mnk[0],
        n=config.mnk[1],
        k=config.mnk[2],
        num_envs=config.num_envs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    train_env = TorchSelfPlayWrapper(base_env)

    obs_shape = (2, config.mnk[0], config.mnk[1])  # (channels, height, width)
    action_dim = config.mnk[0] * config.mnk[1]

    return train_env, obs_shape, action_dim


def create_agent(config, obs_shape, action_dim, device):
    """Initialize PPO agent with network, optimizer, and scheduler."""
    network = create_model_from_architecture(
        config.architecture_name, obs_shape=obs_shape, action_dim=action_dim
    )
    network = network.to(device)

    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.learning_rate,
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
        "learning_rate": 3e-4,
        "lr_warmup_steps": 200_000,
        # entropy
        "entropy_coef": 0.02,
        "entropy_coef_schedule": {
            "type": "linear",
            "params": {"final_coef": 0.001, "total_steps": 30_000_000},
        },
        # ppo
        "gamma": 0.99,
        "clip_range": 0.2,
        "batch_size": 4096,
        "n_steps": 256,
        "ppo_epochs": 4,
        "total_environment_steps": 50_000_000,
        "num_envs": 512,
        # validation
        "benchmark_update_threshold_score": 0.60,
        "validation_interval": 5,
        "validation_episodes": 256,
        # selfplay
        "opponent_pool": 10,
        #
        "architecture_name": "resnet_s",
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with wandb.init(config=default_config, project="mnk_995_gpu") as run:
        model_exporter = ModelExporter(run.name or None)

        train_env, obs_shape, action_dim = setup_environment(run.config)

        agent = create_agent(run.config, obs_shape, action_dim, device)
        run.watch(agent.network)

        benchmark_policy = NNPolicy(deepcopy(agent.network))

        opponent_pool = OpponentPool(max_size=run.config.opponent_pool)
        opponent_pool.add_opponent(NNPolicy(deepcopy(agent.network)))

        steps_per_iteration = run.config.num_envs * run.config.n_steps
        total_iterations = run.config.total_environment_steps // steps_per_iteration
        print(f"Starting training for {total_iterations} iterations")

        current_env_steps = 0
        for i in range(total_iterations):
            try:
                if random.random() < 0.2:
                    opponent = opponent_pool.get_random_opponent()
                else:
                    opponent = NNPolicy(deepcopy(agent.network))
                train_env.set_opponent(opponent)

                metrics = agent.learn(train_env)

                current_env_steps = (i + 1) * steps_per_iteration

                log_training_metrics(
                    run, metrics, i, current_env_steps, agent.entropy_coef, agent
                )

                if i % 20 == 0 and metrics.mean_reward:
                    opponent_pool.add_opponent(NNPolicy(deepcopy(agent.network)))

                if i > 0 and i % run.config.validation_interval == 0:
                    print(
                        f"--- Running validation at step {i} ({current_env_steps:,} env steps) ---"
                    )
                    current_policy = NNPolicy(agent.network)

                    validation_res = validate_gpu(
                        agent_policy=current_policy,
                        opponent_policy=benchmark_policy,
                        mnk_config=run.config.mnk,
                        n_episodes=run.config.validation_episodes,
                        device=device,
                    )

                    agent.network.train()

                    run.log(validation_res, step=current_env_steps)

                    score_rate = validation_res["validation/vs_benchmark/score_rate"]
                    print(f"Validation Score Rate: {score_rate:.2f}")

                    if score_rate > run.config.benchmark_update_threshold_score:
                        print(f"--- New benchmark agent at step {i}! ---")
                        benchmark_policy = NNPolicy(deepcopy(agent.network))

                        model_exporter.export_model(
                            agent.network, i, is_benchmark_breaker=True
                        )

                        run.log({"validation/new_benchmark_step": 1}, step=current_env_steps)
                    else:
                        model_exporter.export_model(
                            agent.network, i, is_benchmark_breaker=False
                        )

            except Exception as e:
                handle_training_error(run, e, i, current_env_steps)
                continue


def log_training_metrics(
    run,
    metrics: TrainingMetrics,
    iteration,
    env_steps,
    entropy_coef,
    agent,
):
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
        f"approx_kl: {metrics.approx_kl:.4f} | "
        f"fps: {metrics.fps:.1f} | "
        f"rollout_time: {metrics.rollout_time:.3f}s | "
        f"learn_time: {metrics.learn_time:.3f}s"
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
