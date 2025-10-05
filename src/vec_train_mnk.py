import torch
import random
from copy import deepcopy
from gymnasium.wrappers.vector import RecordEpisodeStatistics
import wandb
import os
from env.mnk_game_env import create_mnk_env
from selfplay.self_play_wrapper import NNPolicy, RandomPolicy
from selfplay.vector_self_play_wrapper import VectorSelfPlayWrapper, VectorNNPolicy, BatchRandomPolicy
from alg.a2c import A2CAgent, ActorCritic
from validation import validate_episodes


def train_mnk():
    # Define hyperparameters
    config = {
        "mnk": (9, 9, 5),
        "learning_rate": 8e-5,
        "gamma": 0.99,
        "batch_size": 64,
        "n_steps": 1024,
        "hidden_dim": 1024,
        "training_iterations": 1000,
        "validation_interval": 10,
        "validation_episodes": 100,
        "benchmark_update_threshold": 0.55,
        "opponent_pool_size": 10,  # Maximum size of the opponent pool
        "num_envs": 8,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with wandb.init(project="mnk_vector_a2c", config=config, group="vec") as run:
        mnk = run.config.mnk

        # Create initial opponent pool with random policy
        base_env = create_mnk_env(mnk[0], mnk[1], mnk[2])
        random_opponent = BatchRandomPolicy(base_env.action_space("black"))
        
        def env_fn():
            return create_mnk_env(m=mnk[0], n=mnk[1], k=mnk[2])

        # Create vectorized environment using VectorSelfPlayWrapper
        train_env = VectorSelfPlayWrapper(random_opponent, env_fn, n_envs=run.config.num_envs)
        train_env = RecordEpisodeStatistics(train_env)

        # Instantiate the A2CAgent
        obs_shape = train_env.single_observation_space["observation"].shape
        action_dim = train_env.single_action_space.n
        network = ActorCritic(obs_shape, action_dim, hidden_dim=run.config.hidden_dim)
        agent = A2CAgent(obs_shape, action_dim, network, n_steps=run.config.n_steps, learning_rate=run.config.learning_rate,
                         gamma=run.config.gamma, batch_size=run.config.batch_size, device=device, num_envs=run.config.num_envs)

        run.watch(agent.network)

        # Create benchmark agent for validation
        benchmark_policy = NNPolicy(deepcopy(agent.network))

        # Maintain an opponent pool to provide varied opponents during training
        opponent_pool = [random_opponent]  # Start with random opponent

        # Training loop with dynamic opponent pool
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
                step=i,
            )

            # Periodically add current agent to opponent pool to improve training difficulty
            if i > 0 and i % 5 == 0:  # Add new opponent every 5 iterations
                new_opponent = VectorNNPolicy(deepcopy(agent.network), device=device)
                opponent_pool.append(new_opponent)
                
                # Limit pool size by removing oldest opponents
                if len(opponent_pool) > run.config.opponent_pool_size:
                    opponent_pool.pop(0)
                
                # Select a random opponent from the pool for next training steps
                train_env.opponent = random.choice(opponent_pool)
                print(f"  Added new opponent to pool, now size: {len(opponent_pool)}")

            # Occasionally switch to a different opponent from the pool
            elif i > 0 and i % 2 == 0 and len(opponent_pool) > 1:
                train_env.opponent = random.choice(opponent_pool)

            if i > 0 and i % run.config.validation_interval == 0:
                print(f"--- Running validation at step {i} ---")

                validation_res = validate(mnk, run.config.validation_episodes, agent, benchmark_policy, device)
                run.log(validation_res, step=i)

                if validation_res["validation/vs_benchmark/win_rate"] > run.config.benchmark_update_threshold:
                    print(f"--- New benchmark agent at step {i}! ---")
                    benchmark_policy = NNPolicy(deepcopy(agent.network))

                    # Add the new benchmark to opponent pool as well
                    benchmark_vector_policy = VectorNNPolicy(deepcopy(agent.network), device=device)
                    opponent_pool.append(benchmark_vector_policy)
                    if len(opponent_pool) > run.config.opponent_pool_size:
                        opponent_pool.pop(0)

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