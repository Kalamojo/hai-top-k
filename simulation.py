import polars as pl
import numpy as np
import numpy.typing as npt
import pickle
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
import uuid
from mallows import TopKMallows, DiverseTopKMallows


def prep_sushi(
        df, nominal_fields, numerical_fields, drop_fields, 
        num_rows, id_field = "id"
    ):
    df = df.limit(
        num_rows
    ).sort(
        id_field
    )
    arr = df.drop(
        drop_fields
    ).with_columns(
        (pl.col(numerical_fields) - pl.col(numerical_fields).min()) / 
        (pl.col(numerical_fields).max() - pl.col(numerical_fields).min())
    ).with_columns(
        df[nominal_fields].to_dummies(drop_first=True)
    ).to_numpy()

    inds = (df[id_field].sort() + 1).to_list()

    return arr, inds

def path_provider(args: pl.FileProviderArgs):
    assert args.index_in_partition == 0

    return f"Distance_{
            args.partition_keys.cast(pl.String).item()
        }/{
            uuid.uuid4().hex
        }.parquet"

def process_combo(
        human_ranks: list[int], algo_ranks: list[int], distance: int,
        use_human_utility: bool, human_beta: float, algo_beta: float,
        algo_alpha: float, algo_sigma: float, universe: list[int],
        algo_k: int, p_penalty: float, sushi_arr: npt.NDArray[np.float64],
        k: int
    ):
    utility_arr = np.zeros(11)
    if use_human_utility:
        utility_arr[human_ranks[0]] = 1
    else:
        utility_arr[algo_ranks[0]] = 1

    human = TopKMallows(
        center=human_ranks, k=k, beta=human_beta, p=p_penalty
    )
    algo = DiverseTopKMallows(
        center=algo_ranks, k=algo_k, beta=algo_beta, p=p_penalty,
        embeddings=sushi_arr, alpha=algo_alpha, sigma=algo_sigma
    )

    util = human.collab_utility(algo, universe, utility_arr)

    return (
        human_ranks, algo_ranks, util, distance, use_human_utility, human_beta, 
        algo_beta, algo_alpha, algo_sigma
    )

def load_combos(
        combos: product[tuple[int, bool, float, float, float, float]],
        furthest_ranks: dict[float, list[tuple[list[int], list[int]]]],
        universe: list[int], algo_k: int, p_penalty: float, 
        sushi_arr: npt.NDArray[np.float64], k: int = 10
    ):
        print("Creating delay generator")
        for combo in tqdm(combos):
            distance, *rest = combo
            for pair in tqdm(furthest_ranks[distance], leave=False):
                yield delayed(process_combo)(
                    *pair, distance, *rest, universe,
                    algo_k, p_penalty, sushi_arr, k
                )

def main():
    distances_path = "./data/pref_distances.pickle"
    sushi_path = "./data/sushi3.idata"
    utilities_output_path = "./data/diversity_utilities"

    nominal_fields = ["minor_group"]
    numerical_fields = [
        "oiliness", "eaten_frequency", "price", "sold_frequency"
    ]
    drop_fields = ["id", "name", "major_group"] + nominal_fields


    with open(distances_path, 'rb') as file:
        furthest_ranks = pickle.load(file)

    
    sushi_df = pl.read_csv(
        sushi_path, separator="\t", has_header=False, 
        new_columns=[
            "id", "name", "style", "major_group",
            "minor_group", "oiliness", "eaten_frequency",
            "price", "sold_frequency"
        ]
    )

    k = 10
    sushi_arr, sushi_inds = prep_sushi(
        sushi_df, nominal_fields, numerical_fields, drop_fields, k
    )
    del sushi_df

    distances = [0, 9, 18, 27, 36, 45]
    universe = [0] + sushi_inds

    human_utility_flags = [True, False]
    human_betas = [1.5, 0.5]
    algorithm_betas = [1.5]
    algorithm_alphas = [0.85]
    algorithm_sigmas = [1.5]
    p_penalty = 0.321
    algo_k = 2

    chunk_size = 500000

    combos = product(
        distances, human_utility_flags, human_betas, algorithm_betas, 
        algorithm_alphas, algorithm_sigmas
    )
    
    print("Beginning parallel call")
    res_gen = Parallel(n_jobs=-1, return_as="generator", verbose=10)(
        load_combos(
                combos, furthest_ranks, universe,
                algo_k, p_penalty, sushi_arr, k
            )
    )
    print("Parallel call generator created")

    schema = [
        "Human-Ranking", "Algo-Ranking", "Utility", "Distance",
        "Uses-Human-Utility", "Human-Beta", "Algorithm-Beta",
        "Algorithm-Alpha", "Algorithm-Sigma"
    ]
    
    chunk = []
    for res in res_gen:
        chunk.append(res)
        if len(chunk) >= chunk_size:
            df = pl.LazyFrame(chunk, orient="row", schema=schema)
            df.sink_parquet(pl.PartitionBy(
                utilities_output_path,
                key="Distance",
                file_path_provider=path_provider
            ))
            print(f"Chunk of size {len(chunk)} written")
            chunk.clear()

    if len(chunk) > 0:
        df = pl.LazyFrame(chunk, orient="row", schema=schema)
        df.sink_parquet(pl.PartitionBy(
            utilities_output_path,
            key="Distance",
            file_path_provider=path_provider
        ))
        print("Final chunk written and done")


if __name__ == "__main__":
    main()
