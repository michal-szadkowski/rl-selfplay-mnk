import os
from collections import defaultdict
from typing import Dict, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ResultsVisualizer:
    """Creates visualizations for tournament results."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_all_visualizations(self, results: Dict[str, Any]) -> None:
        """Create visualization charts."""
        self.create_elo_progress_chart(results["elo_ratings"])

    def _export_figure(
        self, fig: go.Figure, filename: str, width: int = 800, height: int = 600
    ) -> None:
        """Export figure to both HTML and PNG formats."""
        html_path = os.path.join(self.output_dir, f"{filename}.html")
        fig.write_html(html_path)

        png_path = os.path.join(self.output_dir, f"{filename}.png")
        try:
            fig.write_image(png_path, width=width, height=height, scale=2)
        except Exception:
            pass

    def create_elo_progress_chart(self, elo_ratings) -> None:
        """Create ELO vs iteration summary chart."""
        df = pd.DataFrame(elo_ratings)

        # Sort by iteration for proper line plotting
        df = df.sort_values("iteration")

        fig = go.Figure()

        # Group by run and create separate traces
        run_groups = defaultdict(list)
        for _, row in df.iterrows():
            run_groups[row["run_name"]].append(row)

        colors = px.colors.qualitative.Set1

        for i, (run_name, entries) in enumerate(run_groups.items()):
            entries_df = pd.DataFrame(entries)
            entries_df = entries_df.sort_values("iteration")

            fig.add_trace(
                go.Scatter(
                    x=entries_df["iteration"],
                    y=entries_df["rating"],
                    mode="markers+lines",
                    name=run_name,
                    marker=dict(size=14, color=colors[i % len(colors)]),
                    line=dict(width=2, color=colors[i % len(colors)]),
                    hovertemplate=f"<b>{run_name}</b><br>Iteration: %{{x}}<br>ELO: %{{y:.0f}}<extra></extra>",
                )
            )

        fig.update_layout(
            title="ELO Rating vs Training Iteration",
            xaxis_title="Training Iteration",
            yaxis_title="ELO Rating",
            height=600,
            hovermode="closest",
            font=dict(size=14),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Export to both HTML and PNG
        self._export_figure(fig, "elo_vs_iteration_summary", width=1000, height=600)
