import polars as pl
import polars.selectors as cs
import pickle
from collections import defaultdict
from joblib import Parallel, delayed
from mallows import TopKMallows


def process_pair(
        i: int, j: int, pref_list: list[list[int]],
        algo_k: int, p_penalty: float
    ):
    rank_i = [ind + 1 for ind in pref_list[i]]
    rank_j = [ind + 1 for ind in pref_list[j]]
    dist = TopKMallows.kendall_distance(
        rank_i, rank_j, algo_k, p_penalty
    )
    return (dist, (rank_i, rank_j))

def main():
    sushi_path = "./data/sushi3a.5000.10.order"
    distances_path = "./data/pref_distances.pickle"

    pref_arr = pl.read_csv(
        sushi_path, separator=" ", has_header=False, skip_lines=1
    ).select(
        cs.exclude(cs.by_index([0, 1]))
    ).to_numpy()

    pref_list = pref_arr.tolist()

    ind_pairs = []
    for i in range(len(pref_list)):
        for j in range(i+1, len(pref_list)):
            ind_pairs.append((i, j))

    algo_k = 10
    p_penalty = 0.321

    res = Parallel(n_jobs=-2, verbose=10)(
        delayed(process_pair)(
            *pair, pref_list, algo_k, p_penalty
        ) for pair in ind_pairs
    )

    furthest_ranks = defaultdict(list)
    for key, val in res:
        furthest_ranks[key].append(val)

    with open(distances_path, 'wb') as file:
        pickle.dump(furthest_ranks, file)

if __name__ == "__main__":
    main()
