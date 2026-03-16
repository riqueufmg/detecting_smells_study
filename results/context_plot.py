import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


CSV_PATH = "prompts_context_size.csv"
OUTPUT_DIR = "plots_context_size"


def load_data(csv_path):
    df = pd.read_csv(csv_path, sep=";")

    if "cont" in df.columns:
        df = df.drop(columns=["cont"], errors="ignore")

    required = ["approach", "smell", "context_size"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["approach"] = df["approach"].astype(str).str.strip()
    df["smell"] = df["smell"].astype(str).str.strip()
    df["context_size"] = pd.to_numeric(df["context_size"], errors="coerce")

    df = df.dropna(subset=["approach", "smell", "context_size"])
    df = df[df["context_size"] > 0].copy()

    return df


def plot_context_size_by_smell(df, output_dir):
    sns.set_style("whitegrid")

    smell_order = sorted(df["smell"].unique())
    hue_order = ["raw code", "metrics"]
    present_hues = [h for h in hue_order if h in df["approach"].unique()]

    palette = {
        "raw code": "#1f77b4",
        "metrics": "#ff7f0e"
    }

    plt.figure(figsize=(11, 6))

    ax = sns.boxplot(
        data=df,
        x="smell",
        y="context_size",
        hue="approach",
        order=smell_order,
        hue_order=present_hues,
        palette=palette,
        width=0.7,
        fliersize=0
    )

    sns.stripplot(
        data=df,
        x="smell",
        y="context_size",
        hue="approach",
        order=smell_order,
        hue_order=present_hues,
        palette=palette,
        dodge=True,
        size=2,
        alpha=0.20,
        ax=ax
    )

    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Context Size (tokens, log scale)")
    #ax.set_title("Context Size by Smell and Approach")
    ax.tick_params(axis="x", rotation=15)

    handles, labels = ax.get_legend_handles_labels()

    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)

    wanted_handles = []
    wanted_labels = []
    for label in present_hues:
        if label in unique_labels:
            idx = unique_labels.index(label)
            wanted_handles.append(unique_handles[idx])
            wanted_labels.append(unique_labels[idx])

    ax.legend(
        wanted_handles,
        wanted_labels,
        title="Approach",
        loc="best"
    )

    plt.tight_layout()

    out_file = Path(output_dir) / "context_size_by_smell_and_approach.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out_file)


def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    df = load_data(CSV_PATH)
    plot_context_size_by_smell(df, OUTPUT_DIR)


if __name__ == "__main__":
    main()