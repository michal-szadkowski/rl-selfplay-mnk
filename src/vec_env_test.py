import torch
from copy import deepcopy
from gymnasium.wrappers.vector import RecordEpisodeStatistics
import os
from env.mnk_game_env import create_mnk_env
from selfplay.self_play_wrapper import NNPolicy, RandomPolicy
from selfplay.vector_self_play_wrapper import VectorSelfPlayWrapper, VectorNNPolicy, BatchRandomPolicy
from alg.a2c import A2CAgent, ActorCritic
from validation import validate_episodes


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


def main():
    # Simple configuration
    config = {
        "mnk": (9, 9, 5),
        "learning_rate": 8e-5,
        "gamma": 0.99,
        "batch_size": 64,
        "n_steps": 1024,
        "hidden_dim": 1024,
        "training_iterations": 20,  # Reduced for testing
        "validation_interval": 5,
        "validation_episodes": 50,
        "benchmark_update_threshold": 0.55,
        "num_envs": 4,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    mnk = config["mnk"]

    # Create initial opponent (random policy)
    base_env = create_mnk_env(mnk[0], mnk[1], mnk[2])
    random_opponent = BatchRandomPolicy(base_env.action_space("black"))
    
    def env_fn():
        return create_mnk_env(m=mnk[0], n=mnk[1], k=mnk[2])

    # Create the vector environment
    train_env = VectorSelfPlayWrapper(random_opponent, env_fn, n_envs=config["num_envs"])
    train_env = RecordEpisodeStatistics(train_env)

    # Create the agent
    obs_shape = train_env.single_observation_space["observation"].shape
    action_dim = train_env.single_action_space.n
    network = ActorCritic(obs_shape, action_dim, hidden_dim=config["hidden_dim"])
    agent = A2CAgent(obs_shape, action_dim, network, n_steps=config["n_steps"], 
                     learning_rate=config["learning_rate"], gamma=config["gamma"], 
                     batch_size=config["batch_size"], device=device, num_envs=config["num_envs"])

    # Create benchmark agent for validation
    benchmark_policy = NNPolicy(deepcopy(agent.network))

    print("Starting training with VectorSelfPlayWrapper and dynamic opponent updates...")
    
    # Training loop
    for i in range(config["training_iterations"]):
        mean_reward, mean_length, actor_loss, critic_loss, entropy_loss = agent.learn(train_env)

        print(f"Iteration {i + 1}: Mean reward = {mean_reward:.3f}, Mean length = {mean_length:.1f}")
        print(f"  Actor loss = {actor_loss:.4f}, Critic loss = {critic_loss:.4f}, Entropy loss = {entropy_loss:.4f}")

        # Update opponent after every iteration (true self-play)
        # Only update if we're not at the first iteration to allow initial learning
        if i > 0:
            new_opponent = VectorNNPolicy(deepcopy(agent.network), device=device)
            train_env.opponent = new_opponent
            print(f"  Updated opponent - training against current agent version")
        else:
            print(f"  Using initial random opponent for first iteration")

        # Validation
        if i > 0 and i % config["validation_interval"] == 0:
            print(f"--- Running validation at step {i} ---")
            validation_res = validate(mnk, config["validation_episodes"], agent, benchmark_policy, device)
            
            if validation_res["validation/vs_benchmark/win_rate"] > config["benchmark_update_threshold"]:
                print(f"--- New benchmark agent at step {i}! ---")
                benchmark_policy = NNPolicy(deepcopy(agent.network))
                save_benchmark_model(agent, "vector_test", i)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()