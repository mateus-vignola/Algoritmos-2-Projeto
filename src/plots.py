"""Accessible Plotly figures required by the experimental report."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CATALOG_OPERATIONS = {
    "search_by_id": (
        "01_search_by_id",
        "Busca exata por ID",
        "Localiza um filme por sua chave única.",
        "Busca por ID",
    ),
    "search_by_category": (
        "02_search_by_category",
        "Busca por categoria",
        "Localiza os filmes associados a uma categoria.",
        "Busca por categoria",
    ),
    "search_by_title": (
        "03_search_by_title",
        "Busca pelo nome do filme",
        "Localiza títulos iguais sem perder duplicatas.",
        "Busca por nome",
    ),
    "insert_catalog": (
        "04_insert_catalog",
        "Inserção no catálogo",
        "Adiciona um novo filme ao índice primário.",
        "Inserção",
    ),
    "delete_catalog": (
        "05_delete_catalog",
        "Remoção do catálogo",
        "Remove um filme existente e rebalanceia a árvore quando necessário.",
        "Remoção",
    ),
}

STRUCTURE_COLORS = {"B-tree": "#777777", "B+ tree": "#1769aa"}


def _catalog_random_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Return only the requested catalog operations from the current run."""
    required_columns = {
        "operation", "structure", "order", "sample_size",
        "insertion_order", "time_ms",
    }
    missing_columns = required_columns - set(raw.columns)
    if missing_columns:
        raise ValueError(
            "Resultados incompletos; colunas ausentes: "
            + ", ".join(sorted(missing_columns))
        )

    focused = raw.loc[raw.operation.isin(CATALOG_OPERATIONS)].copy()
    missing_operations = set(CATALOG_OPERATIONS) - set(
        focused["operation"].dropna().astype(str)
    )
    if missing_operations:
        missing = ", ".join(sorted(missing_operations))
        raise ValueError(
            "Os resultados atuais das cinco operações estão incompletos. "
            f"Operações ausentes: {missing}. Execute a célula das operações do catálogo "
            "antes de gerar os gráficos; dados antigos não serão reutilizados."
        )

    focused = focused.loc[focused["insertion_order"].eq("random")].copy()
    if focused.empty:
        raise ValueError("Não há resultados atuais para a ordem de inserção aleatória.")
    return focused


def _mean_operation_times(raw: pd.DataFrame) -> pd.DataFrame:
    focused = _catalog_random_results(raw)
    return focused.groupby(
        ["operation", "structure", "sample_size", "order"], as_index=False
    ).agg(mean_time_ms=("time_ms", "mean"))


def create_figures(raw: pd.DataFrame, summary: pd.DataFrame) -> dict[str, go.Figure]:
    """Create one uncluttered heatmap for each catalog operation."""
    figures: dict[str, go.Figure] = {}
    focused = _catalog_random_results(raw)
    if not focused.empty:
        available_orders = sorted(focused["order"].dropna().astype(int).unique())
        available_sizes = sorted(focused["sample_size"].dropna().astype(int).unique())
        order_labels = [str(order) for order in available_orders]
        size_labels = [
            f"{size // 1_000} mil" if size >= 1_000 else str(size)
            for size in available_sizes
        ]
        for operation, (name, title, explanation, _) in CATALOG_OPERATIONS.items():
            operation_data = focused.loc[focused.operation == operation].copy()
            if operation_data.empty:
                continue
            view = operation_data.groupby(
                ["structure", "sample_size", "order"], as_index=False
            ).agg(
                mean_time_ms=("time_ms", "mean")
            )
            figure = make_subplots(
                rows=1,
                cols=2,
                shared_yaxes=True,
                horizontal_spacing=0.09,
                subplot_titles=("Árvore B", "Árvore B+"),
            )
            for column, structure in enumerate(("B-tree", "B+ tree"), start=1):
                matrix = (
                    view.loc[view["structure"].eq(structure)]
                    .pivot(index="sample_size", columns="order", values="mean_time_ms")
                    .reindex(index=available_sizes, columns=available_orders)
                )
                if matrix.isna().any().any():
                    raise ValueError(
                        f"Resultados incompletos para {operation} / {structure}; "
                        "dados antigos não serão usados para preencher células."
                    )
                minimum = float(matrix.min().min())
                text = [[
                    f"{'★ ' if float(value) == minimum else ''}{float(value):.6f} ms"
                    for value in row
                ] for row in matrix.to_numpy()]
                figure.add_trace(
                    go.Heatmap(
                        x=order_labels,
                        y=size_labels,
                        z=matrix.to_numpy(),
                        text=text,
                        texttemplate="%{text}",
                        coloraxis="coloraxis",
                        hovertemplate=(
                            f"{structure}<br>Ordem: %{{x}}<br>Filmes: %{{y}}"
                            "<br>Tempo médio: %{z:.6f} ms<extra></extra>"
                        ),
                        showscale=False,
                    ),
                    row=1,
                    col=column,
                )
            figure.update_layout(
                title={
                    "text": (
                        f"{title}<br><sup>{explanation} "
                        "Cada célula é uma média; ★ marca a melhor combinação de cada estrutura.</sup>"
                    ),
                    "x": 0.03,
                },
                template="plotly_white",
                font={"family": "Arial", "size": 14},
                margin={"l": 90, "r": 110, "t": 110, "b": 75},
                height=520,
                coloraxis={
                    "colorscale": [
                        [0.0, "#f7fbff"], [0.45, "#9ecae1"], [1.0, "#2171b5"]
                    ],
                    "colorbar": {"title": "Tempo médio<br>(ms)", "thickness": 16},
                },
            )
            figure.update_xaxes(title_text="Ordem da árvore", type="category")
            figure.update_yaxes(title_text="Filmes indexados", type="category", row=1, col=1)
            figures[name] = figure
        if set(figures) != {details[0] for details in CATALOG_OPERATIONS.values()}:
            raise ValueError("Nem todos os cinco gráficos atuais puderam ser criados.")
    return figures


def create_results_overview_figures(raw: pd.DataFrame) -> dict[str, go.Figure]:
    """Create compact comparisons intended for the read-only results notebook."""
    view = _mean_operation_times(raw)
    operations = list(CATALOG_OPERATIONS)
    structures = ["B-tree", "B+ tree"]
    sizes = sorted(view["sample_size"].astype(int).unique())
    largest_size = max(sizes)
    size_labels = [f"{size // 1_000} mil" for size in sizes]
    operation_labels = [CATALOG_OPERATIONS[operation][3] for operation in operations]
    figures: dict[str, go.Figure] = {}

    # A direct answer for the largest workload: two aggregate bars per operation.
    largest = view.loc[view["sample_size"].eq(largest_size)]
    best_largest = largest.loc[
        largest.groupby(["operation", "structure"])["mean_time_ms"].idxmin()
    ]
    best_figure = make_subplots(
        rows=2,
        cols=3,
        horizontal_spacing=0.09,
        vertical_spacing=0.20,
        subplot_titles=tuple(operation_labels) + ("",),
    )
    for panel_index, operation in enumerate(operations):
        subplot_row = panel_index // 3 + 1
        subplot_column = panel_index % 3 + 1
        panel = best_largest.loc[best_largest["operation"].eq(operation)]
        panel_max = float(panel["mean_time_ms"].max())
        for structure in structures:
            row = panel.loc[panel["structure"].eq(structure)].iloc[0]
            value = float(row["mean_time_ms"])
            best_figure.add_trace(
                go.Bar(
                    x=[structure.replace(" tree", "")],
                    y=[value],
                    name=structure,
                    legendgroup=structure,
                    showlegend=panel_index == 0,
                    marker_color=STRUCTURE_COLORS[structure],
                    text=[f"{value:,.6f} ms<br>ordem {int(row['order'])}"],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        f"{CATALOG_OPERATIONS[operation][3]}<br>{structure}"
                        f"<br>Ordem: {int(row['order'])}"
                        f"<br>Tempo médio: {value:,.6f} ms"
                        "<extra></extra>"
                    ),
                ),
                row=subplot_row,
                col=subplot_column,
            )
        best_figure.update_yaxes(
            range=[0, panel_max * 1.28],
            title_text="Tempo médio (ms)" if subplot_column == 1 else None,
            row=subplot_row,
            col=subplot_column,
        )
    best_figure.update_layout(
        title={
            "text": (
                f"Melhor configuração com {largest_size // 1_000} mil filmes"
                "<br><sup>Somente a menor média de cada estrutura é mostrada; menor é melhor.</sup>"
            ),
            "x": 0.03,
        },
        template="plotly_white",
        font={"family": "Arial", "size": 14},
        height=760,
        margin={"l": 80, "r": 35, "t": 120, "b": 70},
        bargap=0.36,
        legend={"orientation": "h", "y": -0.14, "x": 0.5, "xanchor": "center"},
    )
    figures["06_best_largest_size"] = best_figure

    # Fifteen cells summarize which structure wins after choosing its best order.
    best_each_size = view.loc[
        view.groupby(["operation", "structure", "sample_size"])["mean_time_ms"].idxmin()
    ]
    comparison_z: list[list[float]] = []
    comparison_text: list[list[str]] = []
    comparison_custom: list[list[list[float]]] = []
    for operation in operations:
        z_row: list[float] = []
        text_row: list[str] = []
        custom_row: list[list[float]] = []
        for size in sizes:
            point = best_each_size.loc[
                best_each_size["operation"].eq(operation)
                & best_each_size["sample_size"].eq(size)
            ].set_index("structure")
            b_time = float(point.loc["B-tree", "mean_time_ms"])
            bp_time = float(point.loc["B+ tree", "mean_time_ms"])
            b_order = int(point.loc["B-tree", "order"])
            bp_order = int(point.loc["B+ tree", "order"])
            signed_advantage = (b_time - bp_time) / max(b_time, bp_time) * 100
            winner = "B+" if signed_advantage > 0 else "B"
            z_row.append(signed_advantage)
            text_row.append(f"{winner}<br>{abs(signed_advantage):.1f}%")
            custom_row.append([b_time, b_order, bp_time, bp_order])
        comparison_z.append(z_row)
        comparison_text.append(text_row)
        comparison_custom.append(custom_row)
    color_limit = max(abs(value) for row in comparison_z for value in row) or 1
    comparison_figure = go.Figure(go.Heatmap(
        x=size_labels,
        y=operation_labels,
        z=comparison_z,
        zmin=-color_limit,
        zmax=color_limit,
        zmid=0,
        text=comparison_text,
        texttemplate="%{text}",
        customdata=comparison_custom,
        colorscale=[
            [0.0, "#777777"], [0.5, "#f7f7f7"], [1.0, "#1769aa"]
        ],
        colorbar={
            "title": "Vantagem<br>B ← 0 → B+",
            "ticksuffix": "%",
            "thickness": 16,
        },
        hovertemplate=(
            "%{y}<br>Filmes: %{x}<br>"
            "B: %{customdata[0]:,.6f} ms (ordem %{customdata[1]})<br>"
            "B+: %{customdata[2]:,.6f} ms (ordem %{customdata[3]})"
            "<extra></extra>"
        ),
    ))
    comparison_figure.update_layout(
        title={
            "text": (
                "Qual estrutura foi mais rápida?"
                "<br><sup>Cada célula compara a melhor ordem de B com a melhor ordem de B+; "
                "o percentual é a vantagem sobre a estrutura mais lenta.</sup>"
            ),
            "x": 0.03,
        },
        template="plotly_white",
        font={"family": "Arial", "size": 14},
        height=590,
        margin={"l": 150, "r": 120, "t": 115, "b": 65},
    )
    comparison_figure.update_xaxes(title_text="Filmes indexados", type="category")
    figures["07_structure_comparison"] = comparison_figure

    # Fixed best overall order: just two lines in each independently scaled panel.
    best_overall_orders = (
        view.groupby(["operation", "structure", "order"], as_index=False)["mean_time_ms"]
        .mean()
        .loc[lambda frame: frame.groupby(["operation", "structure"])["mean_time_ms"].idxmin()]
    )
    chosen_orders = {
        (row.operation, row.structure): int(row.order)
        for row in best_overall_orders.itertuples()
    }
    scalability_titles = tuple(
        f"{CATALOG_OPERATIONS[operation][3]}"
        f"<br><sup>B: ordem {chosen_orders[(operation, 'B-tree')]} · "
        f"B+: ordem {chosen_orders[(operation, 'B+ tree')]}</sup>"
        for operation in operations
    )
    scalability = make_subplots(
        rows=2,
        cols=3,
        horizontal_spacing=0.09,
        vertical_spacing=0.22,
        subplot_titles=scalability_titles + ("",),
    )
    for panel_index, operation in enumerate(operations):
        subplot_row = panel_index // 3 + 1
        subplot_column = panel_index % 3 + 1
        panel_max = 0.0
        for structure in structures:
            chosen_order = chosen_orders[(operation, structure)]
            line = view.loc[
                view["operation"].eq(operation)
                & view["structure"].eq(structure)
                & view["order"].eq(chosen_order)
            ].sort_values("sample_size")
            values = line["mean_time_ms"].astype(float).tolist()
            panel_max = max(panel_max, max(values))
            scalability.add_trace(
                go.Scatter(
                    x=size_labels,
                    y=values,
                    mode="lines+markers",
                    name=structure,
                    legendgroup=structure,
                    showlegend=panel_index == 0,
                    line={"color": STRUCTURE_COLORS[structure], "width": 3},
                    marker={"size": 8},
                    customdata=[chosen_order] * len(values),
                    hovertemplate=(
                        f"{CATALOG_OPERATIONS[operation][3]}<br>{structure}"
                        "<br>Filmes: %{x}<br>Ordem fixa: %{customdata}"
                        "<br>Tempo médio: %{y:,.6f} ms<extra></extra>"
                    ),
                ),
                row=subplot_row,
                col=subplot_column,
            )
        scalability.update_yaxes(
            range=[0, panel_max * 1.12],
            title_text="Tempo médio (ms)" if subplot_column == 1 else None,
            row=subplot_row,
            col=subplot_column,
        )
        scalability.update_xaxes(
            title_text="Filmes", type="category",
            row=subplot_row, col=subplot_column,
        )
    scalability.update_layout(
        title={
            "text": (
                "Como o tempo cresce com o catálogo"
                "<br><sup>Duas linhas por painel, usando uma ordem fixa: a de menor média geral "
                "para cada estrutura e operação.</sup>"
            ),
            "x": 0.03,
        },
        template="plotly_white",
        font={"family": "Arial", "size": 14},
        height=780,
        margin={"l": 80, "r": 35, "t": 120, "b": 75},
        legend={"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center"},
        hovermode="x unified",
    )
    figures["08_scalability"] = scalability
    return figures


def export_figures(
    figures: dict[str, go.Figure], html_dir: Path, static_dir: Path, export_static: bool
) -> None:
    html_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    for stale in html_dir.glob("*.html"):
        stale.unlink()
    for pattern in ("*.png", "*.svg"):
        for stale in static_dir.glob(pattern):
            stale.unlink()
    for name, figure in figures.items():
        figure.write_html(html_dir / f"{name}.html", include_plotlyjs="cdn")
        if export_static:
            try:
                figure.write_image(static_dir / f"{name}.png", scale=2)
            except (ValueError, RuntimeError) as exc:
                warnings.warn(f"Static export failed for {name}: {exc}", stacklevel=2)
