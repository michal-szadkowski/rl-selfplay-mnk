import math
import pandas as pd


class ELOTracker:
    """ELO rating system with pandas-based processing."""

    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor

    def calculate_ratings(self, match_results: pd.DataFrame) -> pd.DataFrame:
        """Calculate ELO ratings from match results DataFrame."""
        if match_results.empty:
            return pd.DataFrame()

        # Get all unique players
        all_players = set(match_results["player1_unique_id"].unique()) | set(
            match_results["player2_unique_id"].unique()
        )

        # Initialize ratings
        ratings = {player: self.initial_rating for player in all_players}

        # Process matches iteratively until convergence
        for _ in range(50):  # max iterations
            total_change = 0.0
            for _, match in match_results.iterrows():
                p1_id, p2_id = match["player1_unique_id"], match["player2_unique_id"]
                p1_rating, p2_rating = ratings[p1_id], ratings[p2_id]

                # Calculate expected scores and updates
                expected_p1 = self._expected_score(p1_rating, p2_rating)
                change_p1 = self.k_factor * (match["player1_score"] - expected_p1)
                change_p2 = self.k_factor * (match["player2_score"] - (1.0 - expected_p1))

                # Apply changes
                ratings[p1_id] += change_p1
                ratings[p2_id] += change_p2
                total_change += abs(change_p1) + abs(change_p2)

            # Check convergence
            if total_change / (len(match_results) * 2) < 0.1:
                break

        # Calculate player statistics
        player_stats = {}
        for player in all_players:
            as_p1 = match_results[match_results["player1_unique_id"] == player]
            as_p2 = match_results[match_results["player2_unique_id"] == player]

            games = as_p1["total_games"].sum() + as_p2["total_games"].sum()
            wins = as_p1["player1_wins"].sum() + as_p2["player2_wins"].sum()
            draws = as_p1["draws"].sum() + as_p2["draws"].sum()
            losses = as_p1["player2_wins"].sum() + as_p2["player1_wins"].sum()

            # Get model info from first match
            if not as_p1.empty:
                info = as_p1.iloc[0]
                run_name, iteration = info["player1_run_name"], info["player1_iteration"]
            else:
                info = as_p2.iloc[0]
                run_name, iteration = info["player2_run_name"], info["player2_iteration"]

            player_stats[player] = {
                "run_name": run_name,
                "iteration": iteration,
                "games_played": int(games),
                "wins": int(wins),
                "draws": int(draws),
                "losses": int(losses),
                "win_rate": wins / games if games > 0 else 0.0,
            }

        # Create final DataFrame
        final_ratings = []
        for player_id, rating in ratings.items():
            stats = player_stats[player_id]
            final_ratings.append({"unique_id": player_id, "rating": round(rating, 2), **stats})

        return pd.DataFrame(final_ratings).sort_values("rating", ascending=False)

    def _expected_score(self, rating1: float, rating2: float) -> float:
        """Calculate expected score for player 1."""
        return 1.0 / (1.0 + math.pow(10.0, (rating2 - rating1) / 400.0))
