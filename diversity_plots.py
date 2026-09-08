import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt


def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, x.iloc[0], fontweight="bold", color='black',
            ha="left", va="center", transform=ax.transAxes)

def create_plot(df: pl.DataFrame, title: str, path: str):
    g = sns.FacetGrid(
        df, row="Distance", aspect=6, height=2.4,
        row_order=[f"Distance {d}" for d in range(0, 46, 9)],
        sharey=False,
        hue="section", 
        hue_order=[
            "Hβ: 1.5 x Aα: 0.85", "Hβ: 0.5 x Aα: 0.85",
            "Hβ: 1.5 x Aα: 0.0", "Hβ: 0.5 x Aα: 0.0"
        ],
        palette=['mediumseagreen', 'skyblue', 'darkgreen', 'darkblue']
    )

    g.map_dataframe(
        sns.kdeplot, "Utility", fill=True, clip_on=False
    )

    g.map(label, "Distance")

    for ax in g.axes.flat:
        ax.axhline(y=0, lw=2, clip_on=False, color="black")
        ax.set_facecolor("none")

    g.figure.subplots_adjust(hspace=-.15)

    g.set_titles("")
    g.figure.suptitle(title)
    g.set(yticks=[], xlabel="")
    g.set_xlabels("Utility")
    g.despine(bottom=True, left=True)

    g.add_legend(title="Sections")

    plt.savefig(path)

def main():
    diverse_data_path = "./data/diversity_utilities"
    human_sufficient_path = "./plots/human_sufficient.png"
    algo_sufficient_path = "./plots/algorithm_sufficient.png"

    diverse_df = pl.read_parquet(diverse_data_path)

    prepped_diverse_df = diverse_df.with_columns(
        ("Hβ: " + pl.concat_str([
            pl.col("Human-Beta"),
            pl.col("Algorithm-Alpha")
        ], separator=" x Aα: ")).alias("section"),
        (
            "Distance " + pl.col("Distance").cast(pl.String)
        ).alias("Distance")
    )

    create_plot(
        prepped_diverse_df.filter(
            pl.col("Uses-Human-Utility") == True,
        ),
        "Utility for Sufficient Human Views",
        human_sufficient_path
    )

    create_plot(
        prepped_diverse_df.filter(
            pl.col("Uses-Human-Utility") == False,
        ),
        "Utility for Sufficient Algorithm Views",
        algo_sufficient_path
    )

if __name__ == "__main__":
    main()
