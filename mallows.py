import numpy as np
import math
import itertools
from collections import defaultdict

from abc import ABC, abstractmethod
import numpy.typing as npt
from typing import Self


class Mallows(ABC):
    def __init__(self, center: list[int], k: int, beta: float):
        self.center = center
        self.k = k
        self.beta = beta
    
    @staticmethod
    @abstractmethod
    def kendall_distance(center: list[int], tau: tuple[int], k: int, p: float) -> float:
        pass;
    
    def mallows_prob(self, ref_ranking: tuple[int]) -> np.float64:
        distance = self.kendall_distance(self.center, ref_ranking, self.k, self.p)
        return np.exp(-self.beta * distance)
    
    def profile_sampling_probs(self, universe: list) -> dict[frozenset[int], np.float64]:
        profile_lists = list(itertools.permutations(universe, self.k))
        prob_array = np.zeros((len(profile_lists)))
        profile_probs_dict = defaultdict(lambda: 0)

        for i in range(len(profile_lists)):
            prob_array[i] = self.mallows_prob(profile_lists[i])
        
        prob_array /= prob_array.sum()
        for i in range(len(profile_lists)):
            profile_probs_dict[frozenset(profile_lists[i])] += prob_array[i]
        
        return profile_probs_dict
    
    @abstractmethod
    def choice_modeling_probs(self, center: list[int], ref_ranking: frozenset[int], k: int, n: int, beta: float, null_val: int = 0) -> tuple[npt.NDArray[np.float64], npt.NDArray]:
        pass;
    
    def collab_utility(self, recommender: Self, universe: list, utility_arr: npt.NDArray[np.float64]) -> np.float64:
        total_utility = 0
        n = len(universe)
        universe_inds = {universe[i]: i for i in range(n)}

        profile_probs = recommender.profile_sampling_probs(universe)
        for S in profile_probs:
            #print("Set:", S, "prob:", profile_probs[S])
            choice_probs, ordered_items = self.choice_modeling_probs(self.center, S, self.k, n, self.beta)
            #print("Choice probs:", choice_probs, "ordered items:", ordered_items)
            prob_inds = [universe_inds[item] for item in ordered_items]
            #print("Marginal utility:", (choice_probs * utility_arr[prob_inds]).sum(), end='\n\n')
            total_utility += profile_probs[S] * (choice_probs * utility_arr[prob_inds]).sum()
        
        return total_utility
    
    def solo_utility(self, universe: list, utility_arr: npt.NDArray[np.float64]) -> np.float64:
        n = len(universe)
        universe_inds = {universe[i]: i for i in range(n)}

        choice_probs, ordered_items = self.choice_modeling_probs(self.center, frozenset(universe[1:]), n - 1, n, self.beta)
        prob_inds = [universe_inds[item] for item in ordered_items]
        
        return (choice_probs * utility_arr[prob_inds]).sum()

class TopKMallows(Mallows):
    def __init__(self, center: list, k: int, beta: float, p: float, **kwargs):
        super().__init__(center, k, beta, **kwargs)
        self.p = p

    @staticmethod
    def kendall_distance(center: list[int], tau: tuple[int], k: int, p: float) -> float:
        dist = 0
        p_dist = 0
        center_dict = {center[i]: i for i in range(k)}
        tau_dict = {tau[i]: i for i in range(k)}
        remaining_tau = set(tau_dict.keys()) - set(center_dict.keys())

        for i in range(k):
            for j in range(i + 1, k):
                if center[i] not in tau_dict and center[j] not in tau_dict:
                    p_dist += p
                else:
                    if center[i] not in tau_dict:
                        b_later = True
                    elif center[j] not in tau_dict:
                        b_later = False
                    else:
                        b_later = tau_dict[center[i]] > tau_dict[center[j]]
                    a_later = (i > j)
                    dist += a_later != b_later
            for item in remaining_tau:
                if center[i] not in tau_dict:
                    b_later = True
                else:
                    b_later = tau_dict[center[i]] > tau_dict[item]
                a_later = False
                dist += a_later != b_later
        
        p_dist += math.comb(len(remaining_tau), 2) * p
        return dist + p_dist
    
    def profile_sampling_probs(self, universe: list) -> dict[frozenset, np.float64]:
        real_universe = universe[1:]
        profile_lists = list(itertools.permutations(real_universe, self.k))
        prob_array = np.zeros((len(profile_lists)))
        profile_probs_dict = defaultdict(lambda: 0)

        for i in range(len(profile_lists)):
            prob_array[i] = self.mallows_prob(profile_lists[i])
        
        prob_array /= prob_array.sum()
        for i in range(len(profile_lists)):
            profile_probs_dict[frozenset(profile_lists[i])] += prob_array[i]
        
        return profile_probs_dict

    @staticmethod
    def PRIM_POS(j: np.integer, beta: float, item_count: int) -> npt.NDArray[np.float64]:
        return np.exp(-beta * j) / np.exp(-beta * np.arange(1, item_count + 1)).sum(-1)

    @staticmethod
    def PRIM_POS_SEQ(j: int, beta: float, item_count: int, before: bool) -> np.float64:
        if j == 0 and before:
            return 0

        prob_before = TopKMallows.PRIM_POS(np.arange(1, j + 1), beta, item_count).sum(0)
        if before:
            return prob_before
        return 1 - prob_before
    
    def choice_modeling_probs(self, center: list[int], ref_ranking: frozenset[int], k: int, n: int, beta: float, null_val: int = 0) -> tuple[npt.NDArray[np.float64], npt.NDArray]:
        center_set = set(center[:k])
        A_null = ref_ranking.union({null_val})
        A_bar = [a for a in ref_ranking if a not in center_set]
        L = np.array([null_val] + A_bar + center[:k])
        
        r = len(A_bar)
        ell = len(center[:k])
        m = len(L)

        # Reversing
        cur_arr = [(center[i], r + 1 + i) for i in reversed(range(k))]
        DP_table = np.zeros((m, k + 1, ell + 1))

        for j in range(1, k - ell + 1):
            jind = j - 1
            sampled_at_j_prob = 1/(n - k - j + 1)
            none_sampled_prob = 1
            for jp in range(1, j):
                none_sampled_prob *= (1 - (r / (n - k - jp)))
            DP_table[:r+1, jind, 0] = sampled_at_j_prob * none_sampled_prob
        #print("n:", n, "k:", k, "r:", r, "comb numerator:", n - k - (r + 1))
        DP_table[:r+1, k, 0] = (math.comb(n - k - (r + 1), k - ell) 
                                / math.comb(n - k, k - ell)) * (1/(r + 1))
        #print("Initialization complete")
        #print(DP_table)

        for q in range(1, ell + 1):
            a_cur, cur_ind = cur_arr[q - 1]
            max_pos = k - ell + q
            #print("Curs:", a_cur, cur_ind, q)
            if a_cur not in A_null:
                #print("First case")
                DP_table[cur_ind, :, q] = 0
                for j in range(1, k - ell + q + 1):
                    jind = j - 1
                    DP_table[:, jind, q] = (DP_table[:, jind, q-1] * self.PRIM_POS_SEQ(j, beta, max_pos, before=False) 
                                            + DP_table[:, jind-1, q-1] * self.PRIM_POS_SEQ(j-1, beta, max_pos, before=True))
                DP_table[:, k, q] = DP_table[:, k, q-1]
            else:
                #print("Second case")
                for j in range(1, k - ell + q + 1):
                    jind = j - 1
                    DP_table[:, jind, q] = DP_table[:, jind, q-1] * self.PRIM_POS_SEQ(j, beta, max_pos, before=False)
                    DP_table[cur_ind, jind, q] = self.PRIM_POS(j, beta, max_pos) * DP_table[:, jind:, q-1].sum((0, 1))
            #print("Loop for q:", q)
            #print(DP_table)
        
        #print("Total probs:", DP_table.sum((0, 1)))
        item_probs = DP_table[:, :k, ell].sum((1))
        return item_probs, L

class DiverseTopKMallows(TopKMallows):
    def __init__(self, center: list, k: int, beta: float, p: float, embeddings: npt.NDArray[np.float64], alpha: float, sigma: float, **kwargs):
        super().__init__(center, k, beta, p, **kwargs)
        self.alpha = alpha
        self.sigma = sigma
        self.sim_matrix = self.prepare_sim_matrix(embeddings)
        #self.sim_matrix = embeddings @ embeddings.T

    def prepare_sim_matrix(self, embeddings: npt.NDArray[np.float64]):
        D = np.sum(
            (embeddings - embeddings.repeat(embeddings.shape[0], axis=0)
                .reshape((embeddings.shape[0], embeddings.shape[0], embeddings.shape[1]))) ** 2
            , axis=-1
        )
        self.sim_matrix = np.exp(-D / (2 * self.sigma**2))
        return self.sim_matrix

    def subset_det(self, indices: tuple[int], relevance: float) -> np.float64:
        inds_z = [ind - 1 for ind in indices]
        L_s = self.sim_matrix[np.ix_(inds_z, inds_z)]
        L_q = self.alpha * relevance * L_s
        #L_q = relevance * L_s
        
        diag_ind = np.diag_indices_from(L_q)
        L_q[diag_ind] = np.sqrt(relevance)

        return np.linalg.det(L_q)
        #return np.linalg.det(L_s)

    def profile_sampling_probs(self, universe: list) -> dict[frozenset, np.float64]:
        real_universe = universe[1:]
        profile_lists = list(itertools.permutations(real_universe, self.k))
        prob_array = np.zeros((len(profile_lists)))
        profile_probs_dict = defaultdict(lambda: 0)

        for i in range(len(profile_lists)):
            prob_m = self.mallows_prob(profile_lists[i])
            prob_array[i] = self.subset_det(profile_lists[i], prob_m)
            # print("Profile:", profile_lists[i])
            # print("Mallows prob:", prob_m)
            # print("Diverse prob:", prob_array[i])
            # print()
        
        prob_array /= prob_array.sum()
        for i in range(len(profile_lists)):
            profile_probs_dict[frozenset(profile_lists[i])] += prob_array[i]
        
        return profile_probs_dict
