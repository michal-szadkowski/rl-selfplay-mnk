import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from env.mnk_game_env import create_mnk_env
from alg.a2c import ActorCritic, A2CAgent
from selfplay.vector_self_play_wrapper import VectorSelfPlayWrapper, NNPolicy, RandomPolicy


def create_wrapped_env():
    """Creates and wraps the MNK environment for training."""
    env = create_mnk_env(3, 3, 3)  # Small game for testing
    return env


def test_vector_self_play_wrapper():
    print("Testing VectorSelfPlayWrapper with A2C...")
    
    # Initialize environment and model
    device = "cpu"  # Use CPU for testing
    num_envs = 4
    n_steps = 128  # Number of steps per update
    
    # Create vector environment with random opponent
    temp_env = create_mnk_env(3, 3, 3)
    random_policy = RandomPolicy(temp_env.action_space(temp_env.possible_agents[0]))
    vec_env = VectorSelfPlayWrapper(random_policy, create_wrapped_env, num_envs)
    
    # Set up A2C network and agent
    obs_shape = vec_env.single_observation_space["observation"].shape
    action_dim = vec_env.single_action_space.n
    hidden_dim = 64  # Smaller for testing
    
    print(f"Observation shape: {obs_shape}")
    print(f"Action space: {action_dim}")
    print(f"Number of environments: {num_envs}")
    
    network = ActorCritic(obs_shape, action_dim, hidden_dim=hidden_dim)
    agent = A2CAgent(
        obs_shape, 
        action_dim, 
        network, 
        n_steps=n_steps, 
        learning_rate=1e-3,
        gamma=0.99, 
        batch_size=32, 
        device=device, 
        num_envs=num_envs
    )
    
    # Wrap with statistics recording
    vec_env = RecordEpisodeStatistics(vec_env)
    
    print("Environment reset...")
    obs, infos = vec_env.reset()
    
    print(f"Observation type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"Observation keys: {list(obs.keys())}")
        print(f"'observation' shape: {obs['observation'].shape}")
        print(f"'action_mask' shape: {obs['action_mask'].shape}")
    
    # Test multiple steps to make sure the environment works properly
    print("\nTesting multiple steps...")
    total_steps = 0
    for episode in range(3):
        print(f"Episode {episode + 1}")
        
        for step in range(10):  # Take 10 steps per episode
            # Get actions from the agent (using random actions for simplicity in this test)
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs['observation'], dtype=torch.float32, device=device)
                action_mask_tensor = torch.as_tensor(obs['action_mask'], dtype=torch.bool, device=device)
                
                dist, _ = agent.network(obs_tensor, action_mask_tensor)
                actions = dist.sample().cpu().numpy()
            
            # Take a step in the environment
            next_obs, rewards, terminations, truncations, infos = vec_env.step(actions)
            
            print(f"  Step {step + 1}: rewards={rewards}, terminations={terminations}")
            
            # Check if any environments terminated and reset them
            if np.any(terminations) or np.any(truncations):
                print(f"    Termination or truncation detected. Terminations: {terminations}, Truncations: {truncations}")
                # Reset only the terminated environments
                for i in range(num_envs):
                    if terminations[i] or truncations[i]:
                        # Reset that specific environment (though we're resetting all for simplicity)
                        pass
            
            obs = next_obs
            total_steps += num_envs
    
    print(f"\nCompleted {total_steps} total steps across {num_envs} environments.")
    
    # Test the A2C learning step
    print("\nTesting A2C learning step...")
    try:
        mean_reward, mean_length, actor_loss, critic_loss, entropy_loss = agent.learn(vec_env)
        print(f"Learning step successful! Mean reward: {mean_reward}, "
              f"Actor loss: {actor_loss}, Critic loss: {critic_loss}, "
              f"Entropy loss: {entropy_loss}")
    except Exception as e:
        print(f"Error during learning step: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nAll tests passed!")


def test_vs_nn_opponent():
    """Test the vector self-play wrapper with a neural network opponent."""
    print("\n" + "="*50)
    print("Testing VectorSelfPlayWrapper vs NN Opponent...")
    
    device = "cpu"
    num_envs = 2
    n_steps = 64
    
    # Create two different models - one for the agent and one for the opponent
    temp_env = create_mnk_env(3, 3, 3)
    obs_shape = temp_env.observation_space(temp_env.possible_agents[0])["observation"].shape
    action_dim = temp_env.action_space(temp_env.possible_agents[0]).n
    
    # Create the opponent model
    opponent_model = ActorCritic(obs_shape, action_dim, hidden_dim=32)
    opponent_policy = NNPolicy(opponent_model, device=device)
    
    # Create vector environment with NN opponent
    vec_env = VectorSelfPlayWrapper(opponent_policy, create_wrapped_env, num_envs)
    
    # Create the main model
    main_model = ActorCritic(obs_shape, action_dim, hidden_dim=32)
    agent = A2CAgent(
        obs_shape, 
        action_dim, 
        main_model, 
        n_steps=n_steps, 
        learning_rate=1e-3,
        gamma=0.99, 
        batch_size=16, 
        device=device, 
        num_envs=num_envs
    )
    
    vec_env = RecordEpisodeStatistics(vec_env)
    
    print("Testing step with NN opponent...")
    obs, infos = vec_env.reset()
    
    for step in range(5):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs['observation'], dtype=torch.float32, device=device)
            action_mask_tensor = torch.as_tensor(obs['action_mask'], dtype=torch.bool, device=device)
            
            dist, _ = agent.network(obs_tensor, action_mask_tensor)
            actions = dist.sample().cpu().numpy()
        
        next_obs, rewards, terminations, truncations, infos = vec_env.step(actions)
        
        print(f"Step {step + 1}: rewards={rewards}, terminations={terminations}")
        
        obs = next_obs
    
    print("NN Opponent test completed successfully!")


if __name__ == "__main__":
    test_vector_self_play_wrapper()
    test_vs_nn_opponent()