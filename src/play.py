import argparse
import torch
import os
import sys
from env.torch_vector_mnk_env import TorchVectorMnkEnv
from selfplay.policy import Policy, NNPolicy, RandomPolicy
from model_export import load_any_model
from env.constants import PLAYER_WHITE


class HumanPolicy(Policy):
    def __init__(self, env: TorchVectorMnkEnv):
        self.env = env

    def act(self, obs, deterministic: bool = False):
        action_mask = obs["action_mask"][0]  # (Batch=1, Actions) -> (Actions)

        legal_indices = torch.nonzero(action_mask).flatten().cpu().numpy().tolist()

        while True:
            try:
                move_str = input(f"Enter your move (0-{len(action_mask) - 1}): ")
                move = int(move_str)

                if move in legal_indices:
                    return torch.tensor([move], device=self.env.device)
                else:
                    print(f">>> Illegal move {move}. Try: {legal_indices}")
            except ValueError:
                print(">>> Invalid input. Please enter a number.")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting game.")
                sys.exit()


def play_game(env: TorchVectorMnkEnv, p1_policy: Policy, p2_policy: Policy):
    obs = env.reset()
    policies = [p1_policy, p2_policy]  # 0=Black, 1=White

    print("\n--- New Game ---")
    print_board(env)

    done = False
    while not done:
        current_player = env.current_player[0].item()
        policy = policies[current_player]

        ai_obs = {
            "observation": obs["observation"].clone(),
            "action_mask": obs["action_mask"],
        }

        if current_player == PLAYER_WHITE:
            ai_obs["observation"] = torch.flip(ai_obs["observation"], dims=(1,))

        if isinstance(policy, HumanPolicy):
            action = policy.act(obs)
        else:
            with torch.no_grad():
                action = policy.act(ai_obs, deterministic=True)

        obs, reward, done_tensor = env.step(action)

        done = done_tensor[0].item()
        reward_val = reward[0].item()

        player_name = "Black" if current_player == 0 else "White"
        print(f"\n--- Player {player_name} plays move {action.item()} ---")
        print_board(env)

    print("\n--- Game Over ---")

    if reward_val == 1.0:
        winner = "Black" if current_player == 0 else "White"
        print(f"Result: Player {winner} wins!")
    elif reward_val == 0.0:
        print("Result: It's a draw!")
    else:
        print(f"Result: Game ended with reward {reward_val}")


def print_board(env: TorchVectorMnkEnv):
    board = env.boards[0].cpu().numpy()
    m, n = env.m, env.n

    RESET = "\033[0m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    DIM = "\033[90m"
    BOLD = "\033[1m"

    max_idx = m * n - 1
    idx_width = len(str(max_idx))
    cell_width = idx_width + 2

    horizontal_line = "   " + ("+" + "-" * cell_width) * n + "+"

    header = "   "
    for col in range(n):
        header += f" {col:^{cell_width - 1}} "

    print(f"\n{horizontal_line}")

    for row in range(m):
        line_str = f"{row:2d} |"

        for col in range(n):
            idx = row * n + col

            if board[0, row, col] == 1:
                val = f"{RED}{BOLD}X{RESET}"
            elif board[1, row, col] == 1:
                val = f"{BLUE}{BOLD}O{RESET}"
            else:
                val = f"{DIM}{idx}{RESET}"

            if board[0, row, col] == 1 or board[1, row, col] == 1:
                space_left = (cell_width - 1) // 2
                space_right = cell_width - 1 - space_left
                formatted_cell = " " * space_left + val + " " * space_right
            else:
                num_str = str(idx)
                pad = cell_width - len(num_str)
                pad_l = pad // 2
                pad_r = pad - pad_l
                formatted_cell = " " * pad_l + val + " " * pad_r

            line_str += formatted_cell + "|"

        print(line_str)
        print(horizontal_line)
    print()


def main():
    parser = argparse.ArgumentParser(description="Play a game of MNK.")
    parser.add_argument(
        "--p1",
        type=str,
        required=True,
        help="Player 1 (Black): 'human', 'random', or path",
    )
    parser.add_argument(
        "--p2",
        type=str,
        required=True,
        help="Player 2 (White): 'human', 'random', or path",
    )
    parser.add_argument("--m", type=int, default=9)
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = TorchVectorMnkEnv(m=args.m, n=args.n, k=args.k, num_envs=1, device=device)
    action_dim = args.m * args.n

    def load_policy_from_arg(policy_arg: str) -> Policy:
        if policy_arg.lower() == "human":
            return HumanPolicy(env)
        if policy_arg.lower() == "random":
            return RandomPolicy(action_dim)

        try:
            print(f"Loading model from: {policy_arg}")
            import glob

            if os.path.isdir(policy_arg):
                json_files = sorted(glob.glob(os.path.join(policy_arg, "*.json")))
                if not json_files:
                    raise FileNotFoundError(f"No .json metadata in {policy_arg}")
                model_id = os.path.basename(json_files[-1])[:-5]
                model_dir = policy_arg
            else:
                model_dir, filename = os.path.split(policy_arg)
                model_id = filename.replace(".pt", "")

            network = load_any_model(model_dir, model_id, device=device)
            network.eval()

            return NNPolicy(network)

        except Exception as e:
            print(f"Error loading model '{policy_arg}': {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    p1_policy = load_policy_from_arg(args.p1)
    p2_policy = load_policy_from_arg(args.p2)

    print("--- Game Setup ---")
    print(f"Player 1 (Black): {args.p1}")
    print(f"Player 2 (White): {args.p2}")
    print(f"Board: {args.m}x{args.n}, Win: {args.k}")
    print(f"Device: {device}")

    play_game(env, p1_policy, p2_policy)


if __name__ == "__main__":
    main()
