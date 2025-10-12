import math
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MatchResult:
    """Result of a single match between two models."""
    player1_id: str
    player2_id: str
    player1_score: float  # 1.0 for win, 0.5 for draw, 0.0 for loss
    player2_score: float
    game_count: int = 1  # Number of games this result represents


@dataclass
class ELOEntry:
    """ELO rating entry for a model."""
    model_id: str
    rating: float
    games_played: int
    wins: int
    draws: int
    losses: int


class ELOTracker:
    """ELO rating system with batch calculation to avoid order bias."""

    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = {}
        self.match_history: List[MatchResult] = []
        self.game_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            'games': 0, 'wins': 0, 'draws': 0, 'losses': 0
        })

    def add_match(self, player1_id: str, player2_id: str,
                  player1_score: float, player2_score: float, game_count: int = 1):
        """Add a match result to the history."""
        result = MatchResult(
            player1_id=player1_id,
            player2_id=player2_id,
            player1_score=player1_score,
            player2_score=player2_score,
            game_count=game_count
        )
        self.match_history.append(result)

    def calculate_ratings(self) -> Dict[str, ELOEntry]:
        """Calculate final ELO ratings from all matches using batch processing."""
        # Initialize ratings
        ratings = {}
        all_players = set()
        for match in self.match_history:
            all_players.add(match.player1_id)
            all_players.add(match.player2_id)

        for player in all_players:
            ratings[player] = self.initial_rating

        # Process matches in batches to minimize order effects
        # We'll use multiple passes to converge on stable ratings
        converged = False
        iterations = 0
        max_iterations = 10

        while not converged and iterations < max_iterations:
            converged = self._process_batch(ratings)
            iterations += 1

        # Create final entries with stats
        entries = {}
        for player_id, rating in ratings.items():
            stats = self.game_stats[player_id]
            entries[player_id] = ELOEntry(
                model_id=player_id,
                rating=round(rating, 2),
                games_played=stats['games'],
                wins=stats['wins'],
                draws=stats['draws'],
                losses=stats['losses']
            )

        return entries

    def _process_batch(self, ratings: Dict[str, float]) -> bool:
        """Process one batch of matches. Returns True if converged."""
        total_change = 0.0
        match_count = 0

        for match in self.match_history:
            p1_id, p2_id = match.player1_id, match.player2_id
            p1_rating, p2_rating = ratings[p1_id], ratings[p2_id]

            # Calculate expected scores
            expected_p1 = self._expected_score(p1_rating, p2_rating)
            expected_p2 = 1.0 - expected_p1

            # Calculate rating changes
            change_p1 = self.k_factor * (match.player1_score - expected_p1)
            change_p2 = self.k_factor * (match.player2_score - expected_p2)

            # Apply changes
            ratings[p1_id] += change_p1
            ratings[p2_id] += change_p2

            total_change += abs(change_p1) + abs(change_p2)
            match_count += 1

            # Update game statistics
            self.game_stats[p1_id]['games'] += match.game_count
            self.game_stats[p2_id]['games'] += match.game_count

            if match.player1_score > match.player2_score:
                self.game_stats[p1_id]['wins'] += match.game_count
                self.game_stats[p2_id]['losses'] += match.game_count
            elif match.player1_score < match.player2_score:
                self.game_stats[p1_id]['losses'] += match.game_count
                self.game_stats[p2_id]['wins'] += match.game_count
            else:
                self.game_stats[p1_id]['draws'] += match.game_count
                self.game_stats[p2_id]['draws'] += match.game_count

        # Check convergence (average change < 0.1 ELO points)
        if match_count > 0:
            avg_change = total_change / (match_count * 2)
            return avg_change < 0.1

        return True

    def _expected_score(self, rating1: float, rating2: float) -> float:
        """Calculate expected score for player 1."""
        return 1.0 / (1.0 + math.pow(10.0, (rating2 - rating1) / 400.0))

    def get_head_to_head(self, player1_id: str, player2_id: str) -> Dict[str, int]:
        """Get head-to-head statistics between two players."""
        stats = {'player1_wins': 0, 'player2_wins': 0, 'draws': 0, 'total_games': 0}

        for match in self.match_history:
            if match.player1_id == player1_id and match.player2_id == player2_id:
                stats['total_games'] += match.game_count
                if match.player1_score > match.player2_score:
                    stats['player1_wins'] += match.game_count
                elif match.player1_score < match.player2_score:
                    stats['player2_wins'] += match.game_count
                else:
                    stats['draws'] += match.game_count
            elif match.player1_id == player2_id and match.player2_id == player1_id:
                stats['total_games'] += match.game_count
                if match.player1_score > match.player2_score:
                    stats['player2_wins'] += match.game_count
                elif match.player1_score < match.player2_score:
                    stats['player1_wins'] += match.game_count
                else:
                    stats['draws'] += match.game_count

        return stats

  
    def reset(self):
        """Reset all ratings and match history."""
        self.ratings.clear()
        self.match_history.clear()
        self.game_stats.clear()