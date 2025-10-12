#!/usr/bin/env python3
"""
Model Comparison Tool

Compares multiple MNK game models using ELO rating system with tournament management.
Supports loading models from files and folders.

Usage:
    python -m src.compare_models model1.pt model2.pt model3.pt
    python -m src.compare_models models/run1 models/run2
    python -m src.compare_models model1.pt models/run1/ model_00500.pt
"""

import argparse
import sys
import os
from typing import List

from .model_comparison import (
    TournamentManager, TournamentConfig, ModelLoader
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare MNK game models using ELO rating system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare specific model files
  python -m src.compare_models model1.pt model2.pt model3.pt

  # Compare all models from a folder
  python -m src.compare_models models/run1

  # Compare models from multiple folders
  python -m src.compare_models models/run1 models/run2

  # Compare mix of files and folders
  python -m src.compare_models model1.pt models/run1/ model_00500.pt

  # Custom settings
  python -m src.compare_models models/run1 --games 100 --board 7 7 4 --device cuda
        """
    )

    # Input sources - files and/or folders
    parser.add_argument(
        'paths',
        nargs='+',
        help='Model files and/or folders containing models to compare'
    )

    # Tournament configuration
    parser.add_argument(
        '--games', '-g',
        type=int,
        default=50,
        help='Number of games to play between each pair of models (default: 50)'
    )

    parser.add_argument(
        '--board', '-b',
        type=int,
        nargs=3,
        metavar=('M', 'N', 'K'),
        default=[9, 9, 5],
        help='Board dimensions M x N and win condition K (default: 9 9 5)'
    )

    parser.add_argument(
        '--device', '-d',
        choices=['cpu', 'cuda', 'mps'],
        default='cpu',
        help='Device to run models on (default: cpu)'
    )

    parser.add_argument(
        '--output', '-o',
        default='comparison_results',
        help='Output directory for results (default: comparison_results)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Validate model paths
    valid_paths = []
    for path in args.paths:
        if os.path.exists(path) or '*' in path or '?' in path:
            valid_paths.append(path)
        else:
            print(f"Warning: Path not found: {path}")

    if not valid_paths:
        print("Error: No valid model paths found!")
        sys.exit(1)

    # Create tournament configuration
    config = TournamentConfig(
        games_per_pair=args.games,
        device=args.device,
        board_size=tuple(args.board),
        output_dir=args.output
    )

    # Create tournament manager
    tournament = TournamentManager(config)

    # Run tournament
    try:
        results = tournament.run_tournament(valid_paths)

        # Print and save results
        tournament.print_final_results(results)
        output_dir = tournament.save_results(results)

        # Generate visualizations
        from .model_comparison.visualizer import ResultsVisualizer
        visualizer = ResultsVisualizer(output_dir)
        visualizer.create_all_visualizations(results)

        print(f"\nTournament completed successfully!")
        print(f"Results saved to: {output_dir}")

    except KeyboardInterrupt:
        print(f"\nTournament interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during tournament: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()