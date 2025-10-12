import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from itertools import combinations

from .model_loader import ModelLoader, ModelInfo
from .elo_tracker import ELOTracker
from .match_runner import MatchRunner, GameConfig


@dataclass
class TournamentConfig:
    """Configuration for tournament."""
    games_per_pair: int = 50
    device: str = "cpu"
    board_size: Tuple[int, int, int] = (9, 9, 5)  # m, n, k
    output_dir: str = "comparison_results"


class TournamentManager:
    """Manages model comparison tournaments with ELO tracking."""

    def __init__(self, config: TournamentConfig):
        self.config = config
        self.model_loader = ModelLoader(device=config.device)
        self.game_config = GameConfig(
            m=config.board_size[0],
            n=config.board_size[1],
            k=config.board_size[2],
            device=config.device
        )
        self.match_runner = MatchRunner(self.game_config)
        self.elo_tracker = ELOTracker()

        # Store timestamp for output directory creation
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = None  # Will be created when saving results

    def run_tournament(self, model_paths: List[str],
                      progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Run a complete tournament between all models."""
        print(f"Loading models from: {model_paths}")
        models = self.model_loader.load_from_paths(model_paths)

        if len(models) < 2:
            raise ValueError("Need at least 2 models for comparison")

        print(f"Loaded {len(models)} models:")
        for model in models:
            print(f"  - {model.model_id} (run: {model.run_name}, iter: {model.iteration})")

        # Generate all pairs
        model_pairs = list(combinations(models, 2))
        total_pairs = len(model_pairs)
        total_matches = total_pairs * self.config.games_per_pair

        print(f"\nTournament Configuration:")
        print(f"  Models: {len(models)}")
        print(f"  Pairs to play: {total_pairs}")
        print(f"  Games per pair: {self.config.games_per_pair}")
        print(f"  Total games: {total_matches}")
        print(f"  Board size: {self.config.board_size[0]}x{self.config.board_size[1]}, K={self.config.board_size[2]}")
        print(f"  Device: {self.config.device}")
        print(f"  Output directory: {self.config.output_dir}/{self.timestamp}")

        # Track tournament progress
        start_time = time.time()
        completed_pairs = 0
        all_match_results = []

        # Play all pairwise matches
        for model1, model2 in model_pairs:
            completed_pairs += 1

            # Progress tracking before match
            elapsed_time = time.time() - start_time
            if completed_pairs > 1:
                avg_time_per_pair = elapsed_time / (completed_pairs - 1)
                remaining_pairs = total_pairs - (completed_pairs - 1)
                eta_seconds = avg_time_per_pair * remaining_pairs
                eta_hours = eta_seconds // 3600
                eta_minutes = (eta_seconds % 3600) // 60
                eta_seconds = eta_seconds % 60
                progress_percent = ((completed_pairs - 1) / total_pairs) * 100
            else:
                eta_hours = eta_minutes = eta_seconds = 0
                progress_percent = 0

            # Print start of match (without end)
            print(f"{completed_pairs:3d}/{total_pairs} ({progress_percent:5.1f}%) | "
                  f"ETA: {int(eta_hours):02d}:{int(eta_minutes):02d}:{int(eta_seconds):02d} | "
                  f"{model1.model_id} ({model1.run_name}) vs {model2.model_id} ({model2.run_name}) | ", end='', flush=True)

            # Play the match
            p1_score, p2_score, game_stats = self.match_runner.play_match(
                model1, model2, self.config.games_per_pair
            )

            # Add to ELO tracker
            self.elo_tracker.add_match(
                model1.model_id, model2.model_id,
                p1_score, p2_score, self.config.games_per_pair
            )

            # Store match results
            match_result = {
                'player1_id': model1.model_id,
                'player2_id': model2.model_id,
                'player1_run': model1.run_name,
                'player2_run': model2.run_name,
                'player1_iteration': model1.iteration,
                'player2_iteration': model2.iteration,
                'player1_score': p1_score,
                'player2_score': p2_score,
                'games': game_stats
            }
            all_match_results.append(match_result)

            # Print end of match (results)
            p1_wins = game_stats['player1_wins']
            p2_wins = game_stats['player2_wins']
            draws = game_stats['draws']

            print(f"P1: {p1_wins:2d}-{p2_wins:2d}-{draws:2d} | Score: {p1_score:.3f}-{p2_score:.3f}")

            # Call progress callback if provided
            if progress_callback:
                progress_callback(completed_pairs, total_pairs, model1.model_id, model2.model_id)

        # Calculate final ELO ratings
        print(f"\nCalculating final ELO ratings...")
        final_ratings = self.elo_tracker.calculate_ratings()

        # Prepare results
        tournament_time = time.time() - start_time
        results = {
            'tournament_info': {
                'timestamp': self.timestamp,
                'config': self.config,
                'total_models': len(models),
                'total_pairs': total_pairs,
                'total_games': total_matches,
                'tournament_duration': tournament_time
            },
            'models': [
                {
                    'model_id': model.model_id,
                    'run_name': model.run_name,
                    'iteration': model.iteration,
                    'architecture': model.architecture_name
                }
                for model in models
            ],
            'match_results': all_match_results,
            'elo_ratings': [
                {
                    'model_id': model_id,
                    'run_name': next(m.run_name for m in models if m.model_id == model_id),
                    'iteration': next(m.iteration for m in models if m.model_id == model_id),
                    'rating': entry.rating,
                    'games_played': entry.games_played,
                    'wins': entry.wins,
                    'draws': entry.draws,
                    'losses': entry.losses,
                    'win_rate': entry.wins / entry.games_played if entry.games_played > 0 else 0
                }
                for model_id, entry in final_ratings.items()
            ]
        }

        # Sort ELO ratings by rating
        results['elo_ratings'].sort(key=lambda x: x['rating'], reverse=True)

        return results

    def print_final_results(self, results: Dict[str, Any]):
        """Print final tournament results."""
        print(f"\n{'='*80}")
        print(f"TOURNAMENT RESULTS")
        print(f"{'='*80}")

        print(f"\nTotal duration: {results['tournament_info']['tournament_duration']:.1f} seconds")
        print(f"Models: {results['tournament_info']['total_models']}")
        print(f"Games played: {results['tournament_info']['total_games']}")

        print(f"\n{'ELO RANKINGS':^80}")
        print(f"{'-'*80}")
        print(f"{'Rank':<6} {'Model ID':<25} {'Run':<20} {'Iter':<6} {'ELO':<6} {'Games':<6} {'Win%':<6}")
        print(f"{'-'*80}")

        for i, entry in enumerate(results['elo_ratings'], 1):
            win_pct = entry['win_rate'] * 100
            print(f"{i:<6} {entry['model_id']:<25} {entry['run_name']:<20} "
                  f"{entry['iteration']:<6} {entry['rating']:<6.0f} "
                  f"{entry['games_played']:<6} {win_pct:<6.1f}")

        print(f"{'-'*80}")

        # Show model groups by run
        run_groups = {}
        for entry in results['elo_ratings']:
            run_name = entry['run_name']
            if run_name not in run_groups:
                run_groups[run_name] = []
            run_groups[run_name].append(entry)

        if len(run_groups) > 1:
            print(f"\n{'BY RUN GROUP':^80}")
            print(f"{'-'*80}")
            for run_name, entries in run_groups.items():
                avg_rating = sum(e['rating'] for e in entries) / len(entries)
                best_rating = max(e['rating'] for e in entries)
                print(f"{run_name:<20} | Models: {len(entries):<3} | "
                      f"Avg ELO: {avg_rating:<6.0f} | Best: {best_rating:<6.0f}")

    def save_results(self, results: Dict[str, Any]):
        """Save tournament results to files."""
        import json
        import csv
        import os

        # Create output directory only when saving
        self.output_dir = f"{self.config.output_dir}/{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Save full results as JSON
        json_path = os.path.join(self.output_dir, "tournament_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {json_path}")

        # Save ELO ratings as CSV
        csv_path = os.path.join(self.output_dir, "elo_ratings.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Model ID', 'Run Name', 'Iteration', 'ELO Rating',
                           'Games Played', 'Wins', 'Draws', 'Losses', 'Win Rate'])
            for i, entry in enumerate(results['elo_ratings'], 1):
                writer.writerow([
                    i, entry['model_id'], entry['run_name'], entry['iteration'],
                    entry['rating'], entry['games_played'], entry['wins'],
                    entry['draws'], entry['losses'], entry['win_rate']
                ])
        print(f"ELO ratings saved to: {csv_path}")

        # Save match results as CSV
        matches_csv_path = os.path.join(self.output_dir, "match_results.csv")
        with open(matches_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Player 1', 'Player 2', 'Player 1 Score', 'Player 2 Score',
                           'Games Played', 'Player 1 Wins', 'Player 2 Wins', 'Draws'])
            for match in results['match_results']:
                games = match['games']
                writer.writerow([
                    match['player1_id'], match['player2_id'],
                    match['player1_score'], match['player2_score'],
                    games['total_games'], games['player1_wins'],
                    games['player2_wins'], games['draws']
                ])
        print(f"Match results saved to: {matches_csv_path}")

        return self.output_dir