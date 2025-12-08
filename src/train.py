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
from utils.hardware import detect_hardware_config
from utils.model_export import ModelExporter, create_model_from_architecture


def setup_environment(config, hw_config):
    base_env = TorchVectorMnkEnv(
        m=config.mnk[0],
        n=config.mnk[1],
        k=config.mnk[2],
        num_envs=config.num_envs,
        device=hw_config.device,
    )

    train_env = TorchSelfPlayWrapper(base_env)

    obs_shape = (2, config.mnk[0], config.mnk[1])  # (channels, height, width)
    action_dim = config.mnk[0] * config.mnk[1]

    return train_env, obs_shape, action_dim


def create_agent(config, obs_shape, action_dim, hw_config):
    """Initialize PPO agent with network, optimizer, and scheduler."""

    network = create_model_from_architecture(
        config.architecture_name, obs_shape=obs_shape, action_dim=action_dim
    )

    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.learning_rate,
        fused=True,
        eps=1e-5,
    )

    lr_scheduler = create_lr_scheduler(
        optimizer,
        config.lr_warmup_steps,
        config.total_environment_steps,
        config.num_envs,
        config.n_steps,
        decay=config.lr_decay,
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
        hw_config=hw_config,
        n_steps=config.n_steps,
        gamma=config.gamma,
        batch_size=config.batch_size,
        num_envs=config.num_envs,
        ppo_epochs=config.ppo_epochs,
        entropy_coef=config.entropy_coef,
        lr_scheduler=lr_scheduler,
        entropy_scheduler=entropy_scheduler,
        optimizer=optimizer,
        clip_range=config.clip_range,
    )

    return agent


def train_mnk(run):
    hw_config = detect_hardware_config()

    model_exporter = ModelExporter(run.name or None)

    train_env, obs_shape, action_dim = setup_environment(run.config, hw_config)

    agent = create_agent(run.config, obs_shape, action_dim, hw_config)
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
                run.log({"training/opponent_source": "historical"}, step=current_env_steps)
            else:
                opponent = NNPolicy(deepcopy(agent.network))
                run.log({"training/opponent_source": "current_agent"}, step=current_env_steps)
            train_env.set_opponent(opponent)

            metrics = agent.learn(train_env)

            current_env_steps = (i + 1) * steps_per_iteration

            log_training_metrics(run, metrics, i, current_env_steps, agent.entropy_coef, agent)

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
                    device=hw_config.device,
                )

                agent.network.train()

                run.log(validation_res, step=current_env_steps)

                score_rate = validation_res["validation/vs_benchmark/score_rate"]
                win_rate = validation_res["validation/vs_benchmark/win_rate"]
                draw_rate = validation_res["validation/vs_benchmark/draw_rate"]
                loss_rate = validation_res["validation/vs_benchmark/loss_rate"]

                print(
                    f"Score: {score_rate:.2f} | W: {win_rate:.2f} | D: {draw_rate:.2f} | L: {loss_rate:.2f}"
                )

                if score_rate > run.config.benchmark_update_threshold_score:
                    print(f"--- New benchmark agent at step {i}! ---")
                    benchmark_policy = NNPolicy(deepcopy(agent.network))

                    model_exporter.export_model(agent.network, i, is_benchmark_breaker=True)

                    run.log({"validation/new_benchmark_step": 1}, step=current_env_steps)
                else:
                    model_exporter.export_model(agent.network, i, is_benchmark_breaker=False)

        except Exception as e:
            handle_training_error(run, e, i, current_env_steps)
            continue

    model_exporter.export_model(agent.network, total_iterations, is_benchmark_breaker=False)


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


def get_default_config():
    return {
        "mnk": (9, 9, 5),
        # lr
        "learning_rate": 5e-4,
        "lr_warmup_steps": 5_000_000,
        "lr_decay": True,
        # entropy
        "entropy_coef": 0.04,
        "entropy_coef_schedule": {
            "type": "linear",
            "params": {"final_coef": 0.001, "total_steps": 200_000_000},
        },
        # ppo
        "gamma": 0.99,
        "clip_range": 0.2,
        "batch_size": 8192,
        "n_steps": 256,
        "ppo_epochs": 4,
        "total_environment_steps": 300_000_000,
        "num_envs": 1024,
        # validation
        "benchmark_update_threshold_score": 0.60,
        "validation_interval": 5,
        "validation_episodes": 256,
        # selfplay
        "opponent_pool": 15,
        #
        "architecture_name": "resnet_b_s",
    }


if __name__ == "__main__":

    config = get_default_config()

    with wandb.init(
        config=config,
        project="mnk",
        group="main_run_small_board",
        tags=["main_experiment"],
    ) as run:
        train_mnk(run)
