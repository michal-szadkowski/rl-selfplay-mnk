import argparse
import torch

from env.mnk_game_env import create_mnk_env
from env.mnk_game import Color
from alg.ppo import ActorCriticModule
from selfplay.policy import Policy, NNPolicy, RandomPolicy


class HumanPolicy(Policy):
    """Policy that allows a human to play by entering moves in the console."""

    def __init__(self, env):
        self.env = env

    def act(self, obs):
        """Prompts the human for a move and validates it."""
        action_mask = obs["action_mask"]
        legal_moves = [i for i, is_legal in enumerate(action_mask) if is_legal]

        while True:
            try:
                print(f"\nLegal moves are: {legal_moves}")
                move_str = input(
                    f"Enter your move for player '{self.env.agent_selection}' (0 to {len(action_mask) - 1}): "
                )
                move = int(move_str)
                if move in legal_moves:
                    return move
                else:
                    print(">>> Illegal move. Please try again.")
            except ValueError:
                print(">>> Invalid input. Please enter a number.")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting game.")
                exit()


def play_game(env, p1_policy: Policy, p2_policy: Policy):
    """
    Runs a single game between two policies, renders the board at each step,
    and prints the final result.
    """
    env.reset()
    policies = {env.possible_agents[0]: p1_policy, env.possible_agents[1]: p2_policy}

    print("\n--- New Game ---")
    env.render()

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            break

        # Get action from the correct policy
        action = policies[agent].act(obs)
        env.step(action)

        print(f"\n--- Player '{agent}' plays move {action} ---")
        env.render()

    # Announce the final result
    print("\n--- Game Over ---")
    winner = env.game.get_winner()
    if winner is None:
        # Check if the board is full for a draw
        if env.game.turn == env.game.m * env.game.n:
            print("Result: It's a draw!")
        else:
            print("Game ended without a clear winner.")
    elif winner == Color.Black:
        print("Result: Player 1 (Black) wins!")
    elif winner == Color.White:
        print("Result: Player 2 (White) wins!")


def main():
    """
    Main function to parse arguments and start the game.
    Allows playing between models, a human, and a random agent.
    """
    parser = argparse.ArgumentParser(description="Play a game of MNK.")
    parser.add_argument(
        "--p1",
        type=str,
        required=True,
        help="Policy for Player 1 (Black). Can be 'human', 'random', or a path to a .pt model file.",
    )
    parser.add_argument(
        "--p2",
        type=str,
        required=True,
        help="Policy for Player 2 (White). Can be 'human', 'random', or a path to a .pt model file.",
    )
    parser.add_argument("--m", type=int, default=9, help="Board width.")
    parser.add_argument("--n", type=int, default=9, help="Board height.")
    parser.add_argument("--k", type=int, default=5, help="Number of pieces in a row to win.")

    args = parser.parse_args()

    # Create the game environment
    env = create_mnk_env(m=args.m, n=args.n, k=args.k, render_mode="human")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get observation and action space dimensions from the environment
    obs_shape = env.observation_space("black")["observation"].shape
    action_dim = env.action_space("black").n

    def load_policy_from_arg(policy_arg: str) -> Policy:
        """Helper function to load a policy based on the command-line argument."""
        if policy_arg.lower() == "human":
            return HumanPolicy(env)
        if policy_arg.lower() == "random":
            return RandomPolicy(env.action_space("black"))

        # Otherwise, load the policy from a model file
        try:
            print(f"Loading model from: {policy_arg}")
            network = ActorCriticModule(obs_shape, action_dim)
            network.load_state_dict(torch.load(policy_arg, map_location=device))
            return NNPolicy(network, device=device)
        except FileNotFoundError:
            print(f"Error: Model file not found at '{policy_arg}'")
            exit(1)
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)

    # Load policies for both players
    p1_policy = load_policy_from_arg(args.p1)
    p2_policy = load_policy_from_arg(args.p2)

    print("--- Game Setup ---")
    print(f"Player 1 (Black): {args.p1.capitalize()}")
    print(f"Player 2 (White): {args.p2.capitalize()}")
    print(f"Board: {args.m}x{args.n}, Win: {args.k}-in-a-row")
    print(f"Device: {device}")

    # Start the game
    play_game(env, p1_policy, p2_policy)

    env.close()


if __name__ == "__main__":
    main()
