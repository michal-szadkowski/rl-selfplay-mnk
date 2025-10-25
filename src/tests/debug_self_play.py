import numpy as np
from src.selfplay.vector_mnk_self_play import VectorMnkSelfPlayWrapper
from src.tests.test_utils import FirstLegalActionPolicy


def debug_single_environment():
    """Debug single environment to trace step-by-step flow."""
    print("=== DEBUG: Single Environment ===")

    # Create wrapper with single environment
    wrapper = VectorMnkSelfPlayWrapper(m=3, n=3, k=3, n_envs=1)
    opponent = FirstLegalActionPolicy()
    wrapper.set_opponent(opponent)

    print(f"Initial players: {wrapper.players}")
    print(f"Initial agent_selection: {wrapper.envs.agent_selection}")

    # Reset environment
    print("\n--- RESET ---")
    obs, infos = wrapper.reset()
    print(f"After reset - players: {wrapper.players}")
    print(f"After reset - agent_selection: {wrapper.envs.agent_selection}")
    print(f"After reset - opponent moves: {opponent.action_count}")
    print(f"Board state:\n{obs['observation'][0][0]}")  # Black pieces
    print(f"Board state:\n{obs['observation'][0][1]}")  # White pieces
    print(f"Action mask: {obs['action_mask'][0]}")

    # Play several steps with detailed logging
    for step in range(10):
        print(f"\n--- STEP {step + 1} ---")
        print(f"Players: {wrapper.players}")
        print(f"Agent selection: {wrapper.envs.agent_selection}")
        print(f"Autoreset envs: {wrapper._autoreset_envs}")

        # Get legal action
        legal_actions = np.where(obs["action_mask"][0] == 1)[0]
        if len(legal_actions) == 0:
            print("No legal actions available!")
            break

        action = legal_actions[0]
        print(f"Chosen action: {action}")

        # Track opponent moves before step
        opponent_moves_before = opponent.action_count

        # Execute step
        obs, rewards, terminations, truncations, infos = wrapper.step(np.array([action]))

        # Track opponent moves after step
        opponent_moves_after = opponent.action_count
        opponent_moves_this_step = opponent_moves_after - opponent_moves_before

        print(f"Opponent moves this step: {opponent_moves_this_step}")
        print(f"Rewards: {rewards}")
        print(f"Terminations: {terminations}")
        print(f"Truncations: {truncations}")
        print(f"Board state:\n{obs['observation'][0][0]}")
        print(f"Board state:\n{obs['observation'][0][1]}")
        print(f"Action mask: {obs['action_mask'][0]}")

        # Check for double opponent move bug
        if opponent_moves_this_step > 1:
            print("!!! BUG DETECTED: Double opponent move !!!")

        if terminations[0] or truncations[0]:
            print("Game ended!")
            break


def debug_multiple_environments():
    """Debug multiple environments to check consistency."""
    print("\n=== DEBUG: Multiple Environments ===")

    n_envs = 3
    wrapper = VectorMnkSelfPlayWrapper(m=3, n=3, k=3, n_envs=n_envs)
    opponent = FirstLegalActionPolicy()
    wrapper.set_opponent(opponent)

    print(f"Initial players: {wrapper.players}")

    # Reset
    obs, infos = wrapper.reset()
    print(f"After reset - players: {wrapper.players}")
    print(f"After reset - opponent moves: {opponent.action_count}")

    # Play a few steps
    for step in range(5):
        print(f"\n--- STEP {step + 1} ---")
        print(f"Players: {wrapper.players}")
        print(f"Agent selection: {wrapper.envs.agent_selection}")

        # Get actions for each environment
        actions = np.zeros(n_envs, dtype=np.int32)
        for env_idx in range(n_envs):
            legal_actions = np.where(obs["action_mask"][env_idx] == 1)[0]
            if len(legal_actions) > 0:
                actions[env_idx] = legal_actions[0]

        print(f"Actions: {actions}")

        # Track opponent moves
        opponent_moves_before = opponent.action_count

        # Execute step
        obs, rewards, terminations, truncations, infos = wrapper.step(actions)

        opponent_moves_after = opponent.action_count
        print(f"Opponent moves this step: {opponent_moves_after - opponent_moves_before}")
        print(f"Rewards: {rewards}")
        print(f"Terminations: {terminations}")
        print(f"Truncations: {truncations}")

        # Check for autoreset
        if np.any(terminations | truncations):
            print("Some environments terminated - checking autoreset behavior...")
            break


def debug_assertion_bug():
    """Specifically test the assertion bug with None actions."""
    print("\n=== DEBUG: Assertion Bug Test ===")

    wrapper = VectorMnkSelfPlayWrapper(m=3, n=3, k=3, n_envs=2)
    opponent = FirstLegalActionPolicy()
    wrapper.set_opponent(opponent)

    obs, infos = wrapper.reset()

    # Try to trigger the assertion bug
    print("Testing assertion bug...")

    # Create actions array
    actions = np.array([0, 1])

    try:
        obs, rewards, terminations, truncations, infos = wrapper.step(actions)
        print("No assertion error - bug may be fixed or not triggered")
    except AssertionError as e:
        print(f"Assertion error triggered: {e}")
        print("This confirms the bug exists!")
    except Exception as e:
        print(f"Other error: {e}")


def main():
    """Run all debug scenarios."""
    print("Starting VectorMnkSelfPlayWrapper debug session...")
    print("Set breakpoints in this file to step through the execution.")

    debug_single_environment()
    debug_multiple_environments()
    debug_assertion_bug()

    print("\n=== DEBUG COMPLETE ===")
    print("Check the output for:")
    print("1. Double opponent moves (opponent_moves_this_step > 1)")
    print("2. Assertion errors")
    print("3. Inconsistent player assignments")
    print("4. Autoreset issues")


if __name__ == "__main__":
    main()
