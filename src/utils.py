
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def countplot_with_hue(df: pd.DataFrame, x: str, hue: str, title: str | None = None):

    ax = sns.countplot(data=df, x=x, hue=hue)
    if title:
        ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    return ax


def groupby_mean(df: pd.DataFrame, by: list[str], col: str) -> pd.DataFrame:
    return df[by + [col]].groupby(by, as_index=False).mean()


def plot_distribution_pairs(
    df: pd.DataFrame,
    cols: list[str],
    hue: str | None = None,
    bins: int = 30,
    palette: str = "Set2"
):
    for col in cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(
            data=df,
            x=col,
            hue=hue,
            bins=bins,
            kde=True,
            palette=palette if hue else None,
            alpha=0.6
        )
        title = f"{col} Distribution"
        if hue:
            title += f" grouped by {hue}"
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()


def plot_categorical_pairs(
    df: pd.DataFrame,
    cols: list[str],
    hue: str | None = None
):
    import matplotlib.pyplot as plt

    for col in cols:
        title = f"{col} Distribution"
        if hue:
            title += f" grouped by {hue}"
        countplot_with_hue(df, x=col, hue=hue, title=title)
        plt.show()
