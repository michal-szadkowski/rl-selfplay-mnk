import os
from collections import defaultdict
from typing import Dict, Any, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


class ResultsVisualizer:
    """Creates visualizations for tournament results."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_all_visualizations(self, results: Dict[str, Any]) -> None:
        """Create all visualization charts."""
        print(f"\nGenerating visualizations in {self.output_dir}...")

        # 1. ELO Progress by Iteration (scatter plot, X=iteration)
        self.create_elo_progress_chart(results['elo_ratings'])

        # 2. ELO Distribution (histogram)
        self.create_elo_distribution(results['elo_ratings'])

        # 3. Top Models Performance (sorted by ELO)
        self.create_top_models_chart(results['elo_ratings'])

        # 4. Match Results Heatmap
        self.create_match_results_heatmap(results)

        # 5. Tournament Statistics Summary
        self.create_tournament_summary(results)

        # 6. Run Comparison (only if multiple runs)
        if self._has_multiple_runs(results):
            self.create_run_comparison(results)

        print(f"All visualizations saved to {self.output_dir}")

    def _export_figure(self, fig: go.Figure, filename: str, width: int = 800, height: int = 600) -> None:
        """Export figure to both HTML and PNG formats."""
        # HTML export
        html_path = os.path.join(self.output_dir, f"{filename}.html")
        fig.write_html(html_path)

        # PNG export
        png_path = os.path.join(self.output_dir, f"{filename}.png")
        try:
            fig.write_image(png_path, width=width, height=height, scale=2)
            print(f"    PNG: {png_path}")
        except Exception as e:
            print(f"    Warning: Could not export PNG (requires kaleido): {e}")
            print("    Install with: pip install kaleido")

        print(f"  - {filename}: {html_path}")

    def create_elo_progress_chart(self, elo_ratings: List[Dict]) -> None:
        """Create ELO progress by iteration chart (scatter plot)."""
        df = pd.DataFrame(elo_ratings)

        # Sort by iteration for proper line plotting
        df = df.sort_values('iteration')

        fig = go.Figure()

        # Group by run and create separate traces
        run_groups = defaultdict(list)
        for _, row in df.iterrows():
            run_groups[row['run_name']].append(row)

        colors = px.colors.qualitative.Set1

        for i, (run_name, entries) in enumerate(run_groups.items()):
            entries_df = pd.DataFrame(entries)
            entries_df = entries_df.sort_values('iteration')

            fig.add_trace(go.Scatter(
                x=entries_df['iteration'],
                y=entries_df['rating'],
                mode='markers',
                name=run_name,
                marker=dict(size=14, color=colors[i % len(colors)]),
                hovertemplate=f'<b>{run_name}</b><br>Iteration: %{{x}}<br>ELO: %{{y:.0f}}<extra></extra>'
            ))

        fig.update_layout(
            title='ELO Rating Progress by Training Iteration',
            xaxis_title='Training Iteration',
            yaxis_title='ELO Rating',
            height=600,
            hovermode='x unified',
            font=dict(size=14)
        )

        # Export to both HTML and PNG (natural aspect ratio)
        self._export_figure(fig, 'elo_progress_by_iteration', width=900, height=500)

    def create_elo_distribution(self, elo_ratings: List[Dict]) -> None:
        """Create ELO distribution histogram."""
        ratings = [entry['rating'] for entry in elo_ratings]

        fig = go.Figure(data=[
            go.Histogram(
                x=ratings,
                nbinsx=20,
                marker_color='lightblue',
                name='ELO Distribution'
            )
        ])

        fig.update_layout(
            title='Distribution of Final ELO Ratings',
            xaxis_title='ELO Rating',
            yaxis_title='Number of Models',
            height=500
        )

        # Export to both HTML and PNG (natural aspect ratio)
        self._export_figure(fig, 'elo_distribution', width=800, height=500)

    def create_top_models_chart(self, elo_ratings: List[Dict]) -> None:
        """Create top models performance chart (sorted by ELO)."""
        # Get top 10 models by ELO and keep sorted by ELO (best on left)
        top_models = sorted(elo_ratings, key=lambda x: x['rating'], reverse=True)[:10]

        fig = go.Figure()

        # Create bar chart with model names, run and iterations
        model_labels = [f"{m['model_id']}<br>({m['run_name'][:20]}...)" for m in top_models]

        fig.add_trace(go.Bar(
            x=model_labels,
            y=[m['rating'] for m in top_models],
            text=[f"{m['rating']:.0f}" for m in top_models],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>ELO: %{y:.0f}<br>%{customdata}<extra></extra>',
            customdata=[f"{m['run_name']}<br>Iter: {m['iteration']}" for m in top_models]
        ))

        fig.update_layout(
            title='Top 10 Models by ELO Rating',
            xaxis_title='Model (Run, Iteration)',
            yaxis_title='ELO Rating',
            height=600,
            xaxis_tickangle=-45
        )

        # Export to both HTML and PNG (natural aspect ratio)
        self._export_figure(fig, 'top_models_by_elo', width=1000, height=600)

    def create_match_results_heatmap(self, results: Dict[str, Any]) -> None:
        """Create heatmap of match results."""
        match_results = results['match_results']
        elo_ratings = results['elo_ratings']

        # Get all unique models
        models = list(set([r['player1_id'] for r in match_results] +
                         [r['player2_id'] for r in match_results]))

        # Create matrix
        matrix = {}
        for model1 in models:
            matrix[model1] = {}
            for model2 in models:
                if model1 == model2:
                    matrix[model1][model2] = 0.5  # Diagonal
                else:
                    matrix[model1][model2] = None

        # Fill matrix with results
        for match in match_results:
            p1, p2 = match['player1_id'], match['player2_id']
            score1, score2 = match['player1_score'], match['player2_score']

            if matrix[p1][p2] is None:
                matrix[p1][p2] = score1
                matrix[p2][p1] = score2

        # Convert to arrays for heatmap
        z_values = []
        model_ids = sorted(models, key=lambda x:
                          next(e['rating'] for e in elo_ratings if e['model_id'] == x),
                          reverse=True)

        # Create enhanced labels with run and iteration
        def get_model_label(model_id):
            model_info = next(e for e in elo_ratings if e['model_id'] == model_id)
            return f"{model_info['model_id']}<br>({model_info['run_name'][:20]}...<br>iter:{model_info['iteration']})"

        model_labels = [get_model_label(model_id) for model_id in model_ids]

        for model1 in model_ids:
            row = []
            for model2 in model_ids:
                row.append(matrix[model1][model2] if matrix[model1][model2] is not None else 0.5)
            z_values.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=model_labels,
            y=model_labels,
            colorscale='RdBu',
            zmid=0.5,
            zmin=0,
            zmax=1,
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Score: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title='Head-to-Head Match Results Heatmap',
            xaxis_title='Opponent (Run, Iteration)',
            yaxis_title='Model (Run, Iteration)',
            height=700,
            xaxis_tickangle=-45
        )

        # Export to both HTML and PNG (natural aspect ratio)
        self._export_figure(fig, 'match_results_heatmap', width=900, height=700)

    def create_tournament_summary(self, results: Dict[str, Any]) -> None:
        """Create tournament statistics summary."""
        elo_data = results['elo_ratings']
        match_results = results['match_results']
        tournament_info = results['tournament_info']

        # Calculate summary statistics
        top_model = max(elo_data, key=lambda x: x['rating'])
        total_games = sum(m['games']['total_games'] for m in match_results)
        total_matches = len(match_results)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ELO Distribution', 'Top Win Rates',
                          'Run Performance', 'Tournament Summary'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )

        # ELO Distribution
        all_ratings = [e['rating'] for e in elo_data]
        fig.add_trace(
            go.Histogram(x=all_ratings, name='ELO Dist', nbinsx=20, marker_color='lightblue'),
            row=1, col=1
        )

        # Top Models by Win Rate
        top_models_by_winrate = sorted(elo_data, key=lambda x: x['win_rate'], reverse=True)[:10]
        fig.add_trace(
            go.Bar(
                x=[f"{m['model_id']}<br>({m['run_name'][:15]}...)" for m in top_models_by_winrate],
                y=[m['win_rate'] * 100 for m in top_models_by_winrate],
                text=[f"{m['win_rate']*100:.1f}%" for m in top_models_by_winrate],
                textposition='auto',
                name='Top Win Rates',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )

        # Run Performance (only if multiple runs)
        if self._has_multiple_runs({'elo_ratings': elo_data}):
            run_groups = defaultdict(list)
            for entry in elo_data:
                run_groups[entry['run_name']].append(entry)

            run_names = list(run_groups.keys())
            avg_ratings = [sum(e['rating'] for e in run_groups[run]) / len(run_groups[run])
                          for run in run_names]

            fig.add_trace(
                go.Bar(x=run_names, y=avg_ratings, name='Avg by Run', marker_color='lightcoral'),
                row=2, col=1
            )
        else:
            # Single run - show best performing iteration
            iterations = [e['iteration'] for e in elo_data]
            ratings = [e['rating'] for e in elo_data]
            fig.add_trace(
                go.Scatter(x=iterations, y=ratings, mode='markers',
                          name='ELO by Iteration', marker=dict(size=8, color='lightcoral')),
                row=2, col=1
            )

        # Tournament Statistics Table
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                          fill_color='lightblue',
                          align='left'),
                cells=dict(values=[
                    ['Total Models', 'Total Matches', 'Total Games', 'Top Model', 'Top ELO', 'Best Win Rate', 'Duration'],
                    [len(elo_data), total_matches, total_games,
                     top_model['model_id'], f"{top_model['rating']:.0f}",
                     f"{max(e['win_rate'] for e in elo_data)*100:.1f}%",
                     f"{tournament_info['tournament_duration']:.1f}s"]
                ],
                fill_color='lightgray',
                align='left')
            ),
            row=2, col=2
        )

        fig.update_layout(
            title='Tournament Performance Summary',
            height=800,
            showlegend=False
        )

        # Export to both HTML and PNG (natural aspect ratio)
        self._export_figure(fig, 'tournament_summary', width=1000, height=800)

    def create_run_comparison(self, results: Dict[str, Any]) -> None:
        """Create comparison chart between different runs."""
        elo_data = results['elo_ratings']

        # Calculate statistics by run
        run_stats = defaultdict(lambda: {'ratings': [], 'iterations': [], 'best_models': []})
        for entry in elo_data:
            run_stats[entry['run_name']]['ratings'].append(entry['rating'])
            run_stats[entry['run_name']]['iterations'].append(entry['iteration'])

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Run Performance Comparison', 'Best Models per Run'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        run_names = list(run_stats.keys())
        avg_ratings = [sum(run_stats[run]['ratings']) / len(run_stats[run])
                      for run in run_names]
        max_ratings = [max(run_stats[run]['ratings']) for run in run_names]
        model_counts = [len(run_stats[run]['ratings']) for run in run_names]

        # Average ELO by run
        fig.add_trace(
            go.Bar(x=run_names, y=avg_ratings, name='Avg ELO', marker_color='lightblue'),
            row=1, col=1
        )

        # Best ELO by run
        fig.add_trace(
            go.Bar(x=run_names, y=max_ratings, name='Best ELO', marker_color='lightgreen'),
            row=1, col=2
        )

        fig.update_layout(
            title='Multi-Run Comparison Analysis',
            height=400,
            showlegend=True
        )

        # Export to both HTML and PNG (natural aspect ratio)
        self._export_figure(fig, 'run_comparison', width=900, height=400)

    def _has_multiple_runs(self, results: Dict[str, Any]) -> bool:
        """Check if results contain models from multiple runs."""
        run_names = set(entry['run_name'] for entry in results['elo_ratings'])
        return len(run_names) > 1