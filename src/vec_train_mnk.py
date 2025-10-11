import os
from copy import deepcopy

import torch
import wandb
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from alg.ppo import PPOAgent, ActorCriticModule
from env.mnk_game_env import create_mnk_env
from selfplay.policy import NNPolicy, VectorNNPolicy
from selfplay.vector_self_play_wrapper import VectorSelfPlayWrapper
from selfplay.opponent_pool import OpponentPool
from validation import run_validation


def train_mnk():
    # Configuration for wandb sweep - default parameters will be overridden by sweep
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

    # Initialize wandb - if running in sweep mode, config will be overridden by sweep values
    with wandb.init(config=default_config, project="mnk_vector_a2c", group="vec", dir='./wnb') as run:
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
            num_envs=run.config.num_envs
        )

        # Lambda for network creation
        network_creator = lambda: ActorCriticModule(obs_shape, action_dim)
        
        # Initialize opponent pool with new OpponentPool
        opponent_pool = OpponentPool(network_creator, max_size=run.config.opponent_pool_size, device=device)
        opponent_pool.add_opponent(agent.network)

        # Set initial opponent for the training environment
        initial_opponent = opponent_pool.get_latest_opponent()
        train_env.unwrapped.set_opponent(initial_opponent)
        
        run.watch(agent.network)

        # Create benchmark agent for validation
        benchmark_policy = NNPolicy(deepcopy(agent.network))

        # Main training loop
        for i in range(run.config.training_iterations):
            try:
                # Add agent to pool periodically
                if i % 2 == 0:
                    opponent_pool.add_opponent(agent.network)
                
                new_opponent = opponent_pool.sample_opponent()
                if new_opponent:
                    train_env.unwrapped.set_opponent(new_opponent)

                mean_reward, mean_length, actor_loss, critic_loss, entropy_loss = agent.learn(train_env)
    
                print(f"Iteration {i}: Mean reward = {mean_reward:.3f}, Mean length = {mean_length:.1f}")
    
                run.log(
                    {
                        "training/mean_reward": mean_reward,
                        "training/mean_length": mean_length,
                        "training/actor_loss": actor_loss,
                        "training/critic_loss": critic_loss,
                        "training/entropy_loss": entropy_loss,
                    },
                    step=i
                )

                # Validate agent performance periodically
                if i > 0 and i % run.config.validation_interval == 0:
                    print(f"--- Running validation at step {i} ---")
    
                    validation_res = run_validation(mnk, run.config.validation_episodes, agent, benchmark_policy, device, seed=i)
                    run.log(validation_res, step=i)
    
                    # Update benchmark if current agent performs better
                    if validation_res["validation/vs_benchmark/win_rate"] > run.config.benchmark_update_threshold:
                        print(f"--- New benchmark agent at step {i}! ---")
                        benchmark_policy = NNPolicy(deepcopy(agent.network))
    
                        opponent_pool.add_opponent(agent.network)
    
                        save_benchmark_model(agent, run.name or "", i)
    
                        run.log({"validation/new_benchmark_step": i}, step=i)
    
            except Exception as e:
                error_msg = f"Error in iteration {i}: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                run.log({
                    "error/iteration": i,
                    "error/message": str(e),
                    "error/traceback": traceback.format_exc()
                }, step=i)
                # Continue to next iteration or break if critical
                continue


def save_benchmark_model(agent, name, step):
    """Saves the agent's network as a new benchmark model."""
    dirname, _ = os.path.split(os.path.abspath(__file__))
    model_dir = os.path.join(dirname, "models", name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"benchmark_step_{step}.pt")
    torch.save(agent.network.state_dict(), model_path)
    print(f"Saved new benchmark model to {model_path}")


if __name__ == "__main__":
    train_mnk()
