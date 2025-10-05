from gymnasium.wrappers.vector import RecordEpisodeStatistics
import numpy as np

from env.mnk_game_env import create_mnk_env
from alg.a2c import ActorCritic
from selfplay.vector_self_play_wrapper import VectorSelfPlayWrapper, NNPolicy, RandomPolicy


def create_wrapped_env():
    env = create_mnk_env(3, 3, 3)
    return env


def test_vector_self_play_wrapper():
    print("Testing VectorSelfPlayWrapper...")
    
    # Test with a random opponent policy
    temp_env = create_mnk_env(3, 3, 3)
    agent_name = temp_env.possible_agents[0]  # Get the first possible agent name
    vec = VectorSelfPlayWrapper(RandomPolicy(temp_env.action_space(agent_name)), create_wrapped_env, 4)
    
    print(f"Number of environments: {vec.num_envs}")
    print(f"Observation space: {vec.observation_space}")
    print(f"Action space: {vec.action_space}")
    print(f"Single observation space: {vec.single_observation_space}")
    print(f"Single action space: {vec.single_action_space}")
    
    # Test reset
    print("\nTesting reset...")
    obs, infos = vec.reset()
    print(f"Observation type: {type(obs)}")
    print(f"Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
    if isinstance(obs, dict):
        print(f"'observation' shape: {obs['observation'].shape}")
        print(f"'action_mask' shape: {obs['action_mask'].shape}")
    # Print first element of batch to see individual observation structure
    if isinstance(obs, dict):
        first_obs = {key: obs[key][0] for key in obs.keys()}
        print(f"First observation keys: {list(first_obs.keys())}")
    print("Reset successful!")
    
    # Test step
    print("\nTesting step method...")
    obs_shape = vec.single_observation_space["observation"].shape
    action_dim = vec.single_action_space.n
    model = ActorCritic(obs_shape, action_dim)
    
    # Replace opponent with NNPolicy
    vec.opponent = NNPolicy(model)
    
    # Generate random actions for the agent
    actions = [vec.single_action_space.sample() for _ in range(vec.num_envs)]
    print(f"Actions taken: {actions}")
    
    # Perform a step
    next_obs, rewards, terminations, truncations, infos = vec.step(actions)
    
    print(f"Next observation type: {type(next_obs)}")
    if isinstance(next_obs, dict):
        print(f"'observation' shape: {next_obs['observation'].shape}")
        print(f"'action_mask' shape: {next_obs['action_mask'].shape}")
    print(f"Rewards: {rewards}")
    print(f"Terminations: {terminations}")
    print(f"Truncations: {truncations}")
    print(f"Infos keys: {list(infos.keys()) if infos else 'None'}")
    print("Step successful!")
    
    # Test multiple steps to ensure the environment works properly
    print("\nTesting multiple steps...")
    for step in range(5):
        actions = [vec.single_action_space.sample() for _ in range(vec.num_envs)]
        next_obs, rewards, terminations, truncations, infos = vec.step(actions)
        print(f"Step {step + 1}: rewards={rewards}, terminations={terminations}")
        
        # Reset environments that are terminated
        if np.any(terminations):
            # Find terminated environments and reset them
            terminated_indices = np.where(terminations)[0]
            print(f"Terminated environments: {terminated_indices}")
            # Reset the entire vector environment for simplicity
            vec.reset()
    
    # Test with RecordEpisodeStatistics wrapper
    print("\nTesting with RecordEpisodeStatistics wrapper...")
    vec_with_stats = RecordEpisodeStatistics(vec)
    obs, infos = vec_with_stats.reset()
    
    # Run a few episodes to test statistics
    for episode in range(3):
        terminated = np.zeros(vec_with_stats.num_envs, dtype=bool)
        step_count = 0
        
        while not np.all(terminated) and step_count < 20:  # Limit steps to avoid infinite games
            actions = [vec_with_stats.single_action_space.sample() for _ in range(vec_with_stats.num_envs)]
            obs, rewards, terminations, truncations, infos = vec_with_stats.step(actions)
            terminated = np.logical_or(terminations, truncations)
            step_count += 1
            
            if step_count % 5 == 0:
                print(f"Episode {episode + 1}, Step {step_count}")
                print(f"  Rewards: {rewards}")
                print(f"  Terminations: {terminations}")
                print(f"  Truncations: {truncations}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_vector_self_play_wrapper()
