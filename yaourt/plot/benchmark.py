"""Benchmark plotting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas


def hhistograms(
    df: pandas.DataFrame,
    keys: Union[str, Tuple[str, ...]] = "name",
    metric: str = "average",
    baseline: str = "baseline",
    title: str = "Benchmark",
    limit: int = 50,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """
    Plots horizontal histograms with error bars for benchmark comparisons.

    Shows the *limit* best-performing configurations alongside the baseline.
    Inspired by ``onnx_extended.plotting.benchmark.hhistograms``.

    :param df: DataFrame containing benchmark results; must have columns
        matching *keys* and *metric*, plus ``min_exec`` and ``max_exec``
        columns when they exist (used for the x-axis limits)
    :param keys: column name(s) to group by; the last key identifies the
        baseline row via *baseline*
    :param metric: column name holding the primary performance metric
    :param baseline: substring used to identify the baseline row in the last
        *keys* column
    :param title: chart title
    :param limit: maximum number of non-baseline rows to display
    :param ax: existing matplotlib axis; a new figure is created when *None*
    :return: the matplotlib axis

    .. plot::

        import pandas
        from yaourt.plot._data import hhistograms_data
        from yaourt.plot.benchmark import hhistograms

        df = pandas.DataFrame(hhistograms_data())
        hhistograms(df, keys=("input", "name"))
    """
    import pandas

    if not isinstance(keys, (tuple, list)):
        keys = (keys,)

    dfm = (
        df[[*keys, metric]].groupby(list(keys), as_index=False).agg(["mean", "min", "max"]).copy()
    )
    if dfm.shape[1] == 3:
        dfm = dfm.reset_index(drop=False)
    dfm.columns = [*keys, metric, "min", "max"]
    dfi = dfm.sort_values(metric).reset_index(drop=True)
    base = dfi[dfi[keys[-1]].str.contains(baseline)]
    not_base = dfi[~dfi[keys[-1]].str.contains(baseline)].reset_index(drop=True)
    if not_base.shape[0] > limit:
        not_base = not_base[:limit]
    merged = pandas.concat([base, not_base], axis=0)
    merged = merged.sort_values(metric).reset_index(drop=True).set_index(list(keys))

    fig = None
    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, max(1, merged.shape[0] / 2)))

    err_min = merged[metric] - merged["min"]
    err_max = merged["max"] - merged[metric]
    merged[[metric]].plot.barh(ax=ax, title=title, xerr=[err_min, err_max])
    b = df.loc[df[keys[-1]] == baseline, metric].mean()
    ax.plot([b, b], [0, df.shape[0]], "r--")
    x_min = df[metric].min()
    x_max = df[metric].max()
    if "min_exec" in df.columns:
        x_min = (df["min_exec"].min() + x_min) / 2
    ax.set_xlim([x_min, x_max])

    if fig is not None:
        fig.tight_layout()
    return ax
