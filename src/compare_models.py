import argparse
import sys
import os
from datetime import datetime

from model_comparison.model_loader import ModelLoader
from model_comparison.match_runner import MatchRunner, GameConfig
from model_comparison.elo_tracker import ELOTracker
from model_comparison.visualizer import ResultsVisualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare MNK game models using ELO rating system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare specific model files
  uv run src/compare_models.py model1.pt model2.pt model3.pt

  # Compare all models from a folder
  uv run src/compare_models.py models/run1

  # Compare models from multiple folders
  uv run src/compare_models.py models/run1 models/run2

  # Compare mix of files and folders
  uv run src/compare_models.py model1.pt models/run1/ model_00500.pt

  # Custom settings
  uv run src/compare_models.py models/run1 --games 100 --board 7 7 4 --device cuda
        """,
    )

    parser.add_argument(
        "paths", nargs="+", help="Model files and/or folders containing models to compare"
    )

    parser.add_argument(
        "--games",
        "-g",
        type=int,
        default=50,
        help="Number of games to play between each pair of models (default: 50)",
    )

    parser.add_argument(
        "--board",
        "-b",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        default=[9, 9, 5],
        help="Board dimensions M x N and win condition K (default: 9 9 5)",
    )

    parser.add_argument(
        "--device",
        "-d",
        choices=["cpu", "cuda", "mps"],
        default="cuda",
        help="Device to run models on (default: cpu)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="comparison_results",
        help="Output directory for results (default: comparison_results)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    valid_paths = [p for p in args.paths if os.path.exists(p) or "*" in p or "?" in p]
    if not valid_paths:
        print("Error: No valid model paths found!")
        sys.exit(1)

    print("Loading models...")
    models = ModelLoader(device=args.device).load_from_paths(valid_paths)
    if len(models) < 2:
        print("Error: Need at least 2 models for comparison")
        sys.exit(1)

    print(f"Loaded {len(models)} models:")
    for model in models:
        print(f"  - {model.unique_id}")

    game_config = GameConfig(
        m=args.board[0], n=args.board[1], k=args.board[2], device=args.device
    )
    print(f"\nStarting tournament with {len(models)} models...")

    match_results = MatchRunner(game_config).run_tournament_batched(models, args.games)
    if match_results.empty:
        print("No matches were played!")
        sys.exit(1)

    elo_ratings = ELOTracker().calculate_ratings(match_results)

    output_dir = save_results(args.output, elo_ratings, match_results, args)

    ResultsVisualizer(output_dir).create_all_visualizations(
        {
            "elo_ratings": elo_ratings.to_dict("records"),
            "match_results": match_results.to_dict("records"),
        }
    )


def save_results(output_dir_base, elo_ratings, match_results, args):
    """Save tournament results to CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir_base}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    elo_ratings.to_csv(f"{output_dir}/elo_ratings.csv", index=False)
    match_results.to_csv(f"{output_dir}/match_results.csv", index=False)

    return output_dir


if __name__ == "__main__":
    main()
