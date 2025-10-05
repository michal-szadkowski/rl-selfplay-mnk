import torch
import random
from copy import deepcopy
from gymnasium.wrappers.vector import RecordEpisodeStatistics
import wandb
import os
from env.mnk_game_env import create_mnk_env
from selfplay.self_play_wrapper import NNPolicy, RandomPolicy
from selfplay.vector_self_play_wrapper import VectorSelfPlayWrapper, VectorNNPolicy, BatchRandomPolicy
from alg.a2c import A2CAgent, ActorCriticModule
from validation import validate_episodes


def cleanup_opponent_pool(opponent_pool, max_size, device):
    """Remove oldest opponents to maintain pool size and free GPU memory."""
    if len(opponent_pool) > max_size:
        # Remove oldest opponents to maintain maximum size
        excess_count = len(opponent_pool) - max_size
        for _ in range(excess_count):
            removed_opponent = opponent_pool.pop(0)
            # Explicitly delete the removed opponent to free memory
            del removed_opponent
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def create_opponent_from_state_dict(state_dict, obs_shape, action_dim, device):
    """Create a VectorNNPolicy from a state dict, loading to the appropriate device."""
    network = ActorCriticModule(obs_shape, action_dim)
    network.load_state_dict(state_dict)
    network.to(device)
    return VectorNNPolicy(network, device=device)


def add_agent_to_pool(agent, opponent_pool):
    """Add current agent to the opponent pool as a state dict on CPU."""
    new_opponent_state = deepcopy(agent.network.state_dict())
    # Move to CPU to save GPU memory
    new_opponent_state_cpu = {k: v.cpu() for k, v in new_opponent_state.items()}
    opponent_pool.append(new_opponent_state_cpu)


def select_opponent_from_pool(opponent_pool, obs_shape, action_dim, device):
    """Select a random opponent from the pool and return it."""
    # Find state dict opponents in the pool
    state_dict_opponents = [op for op in opponent_pool if isinstance(op, dict)]
    if state_dict_opponents:
        selected_state = random.choice(state_dict_opponents)
        return create_opponent_from_state_dict(
            selected_state, obs_shape, action_dim, device
        )
    return None


def train_mnk():
    config = {
        "mnk": (9, 9, 5),
        "learning_rate": 4e-5,
        "gamma": 0.99,
        "batch_size": 64,
        "n_steps": 512,
        "training_iterations": 1000,
        "validation_interval": 5,
        "validation_episodes": 50,
        "benchmark_update_threshold": 0.65,
        "opponent_pool_size": 10,  # Maximum size of the opponent pool
        "num_envs": 32,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with wandb.init(project="mnk_vector_a2c", config=config, group="vec", dir = './wnb') as run:
        mnk = run.config.mnk

        def env_fn():
            return create_mnk_env(m=mnk[0], n=mnk[1], k=mnk[2])

        # Initialize vectorized self-play environment
        train_env = VectorSelfPlayWrapper(env_fn, n_envs=run.config.num_envs)
        
        obs_shape = train_env.single_observation_space["observation"].shape
        action_dim = train_env.single_action_space.n

        # Initialize A2C agent with neural network
        network = ActorCriticModule(obs_shape, action_dim)
        agent = A2CAgent(obs_shape, action_dim, network, n_steps=run.config.n_steps, learning_rate=run.config.learning_rate,
                         gamma=run.config.gamma, batch_size=run.config.batch_size, device=device, num_envs=run.config.num_envs)

        # Initialize opponent pool with initial agent version
        opponent_pool = []
        add_agent_to_pool(agent, opponent_pool)
        
        initial_opponent_for_env = create_opponent_from_state_dict(
            opponent_pool[0], obs_shape, action_dim, device
        )

        # Set initial opponent for the training environment
        train_env.opponent = initial_opponent_for_env
        
        train_env = RecordEpisodeStatistics(train_env)

        run.watch(agent.network)

        # Create benchmark agent for validation
        benchmark_policy = NNPolicy(deepcopy(agent.network))

        # Main training loop
        for i in range(run.config.training_iterations):
            mean_reward, mean_length, actor_loss, critic_loss, entropy_loss = agent.learn(train_env)

            print(f"Iteration {i + 1}: Mean reward = {mean_reward:.3f}, Mean length = {mean_length:.1f}")

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

            # Update opponent pool with current agent every 5 iterations
            if i > 0 and i % 5 == 0:
                add_agent_to_pool(agent, opponent_pool)
                
                cleanup_opponent_pool(opponent_pool, run.config.opponent_pool_size, device)
                
                # Select and set a new opponent for training
                selected_opponent = select_opponent_from_pool(opponent_pool, obs_shape, action_dim, device)
                if selected_opponent:
                    train_env.opponent = selected_opponent
                print(f"  Added new opponent to pool, now size: {len(opponent_pool)}")

            # Switch to different opponent occasionally
            elif i > 0 and i % 2 == 0:
                selected_opponent = select_opponent_from_pool(opponent_pool, obs_shape, action_dim, device)
                if selected_opponent:
                    train_env.opponent = selected_opponent

            # Validate agent performance periodically
            if i > 0 and i % run.config.validation_interval == 0:
                print(f"--- Running validation at step {i} ---")

                validation_res = validate(mnk, run.config.validation_episodes, agent, benchmark_policy, device)
                run.log(validation_res, step=i)

                # Update benchmark if current agent performs better
                if validation_res["validation/vs_benchmark/win_rate"] > run.config.benchmark_update_threshold:
                    print(f"--- New benchmark agent at step {i}! ---")
                    benchmark_policy = NNPolicy(deepcopy(agent.network))

                    add_agent_to_pool(agent, opponent_pool)
                    
                    # Clean up the pool after adding benchmark
                    cleanup_opponent_pool(opponent_pool, run.config.opponent_pool_size, device)

                    save_benchmark_model(agent, run.name, i)

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
    print(f"Win rate vs random = {random_stats['win_rate']:.3f}/{random_stats['draw_rate']:.3f}")

    # 2. Validate against the benchmark agent
    benchmark_stats = validate_episodes(
        validation_env,
        current_policy,
        benchmark_policy,
        n_episodes,
    )
    print(f"Win rate vs benchmark = {benchmark_stats['win_rate']:.3f}/{benchmark_stats['draw_rate']:.3f}")

    agent.network.train()
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