import numpy as np
import math
import itertools
from collections import defaultdict

from abc import ABC, abstractmethod
import numpy.typing as npt
from typing import Self


class Mallows(ABC):
    def __init__(self, center: list, k: int, beta: float):
        self.center = center
        self.k = k
        self.beta = beta
    
    @abstractmethod
    def kendall_distance(self, tau: tuple) -> float:
        pass;
    
    def mallows_prob(self, ref_ranking: tuple) -> np.float64:
        distance = self.kendall_distance(ref_ranking)
        return np.exp(-self.beta * distance)
    
    def profile_sampling_probs(self, universe: list) -> dict[frozenset, np.float64]:
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
    def choice_modeling_probs(self, ref_ranking: frozenset, k: int, n: int, null_val: int = 0) -> tuple[npt.NDArray[np.float64], npt.NDArray]:
        pass;
    
    def collab_utility(self, recommender: Self, universe: list, utility_arr: npt.NDArray[np.float64]) -> np.float64:
        total_utility = 0
        n = len(universe)
        universe_inds = {universe[i]: i for i in range(n)}

        profile_probs = recommender.profile_sampling_probs(universe)
        for S in profile_probs:
            #print("Set:", S, "prob:", profile_probs[S])
            choice_probs, ordered_items = self.choice_modeling_probs(S, self.k, n)
            #print("Choice probs:", choice_probs, "ordered items:", ordered_items)
            prob_inds = [universe_inds[item] for item in ordered_items]
            #print("Marginal utility:", (choice_probs * utility_arr[prob_inds]).sum(), end='\n\n')
            total_utility += profile_probs[S] * (choice_probs * utility_arr[prob_inds]).sum()
        
        return total_utility
    
    def solo_utility(self, universe: list, utility_arr: npt.NDArray[np.float64]) -> np.float64:
        n = len(universe)
        universe_inds = {universe[i]: i for i in range(n)}

        choice_probs, ordered_items = self.choice_modeling_probs(frozenset(universe[1:]), n - 1, n)
        prob_inds = [universe_inds[item] for item in ordered_items]
        
        return (choice_probs * utility_arr[prob_inds]).sum()

class TopKMallows(Mallows):
    def __init__(self, center: list, k: int, beta: float, p: float, **kwargs):
        self.p = p
        super().__init__(center, k, beta, **kwargs)
    
    def kendall_distance(self, tau: tuple) -> float:
        dist = 0
        p_dist = 0
        center_dict = {self.center[i]: i for i in range(self.k)}
        tau_dict = {tau[i]: i for i in range(self.k)}
        remaining_tau = set(tau_dict.keys()) - set(center_dict.keys())

        for i in range(self.k):
            for j in range(i + 1, self.k):
                if self.center[i] not in tau_dict and self.center[j] not in tau_dict:
                    p_dist += self.p
                else:
                    if self.center[i] not in tau_dict:
                        b_later = True
                    elif self.center[j] not in tau_dict:
                        b_later = False
                    else:
                        b_later = tau_dict[self.center[i]] > tau_dict[self.center[j]]
                    a_later = (i > j)
                    dist += a_later != b_later
            for item in remaining_tau:
                if self.center[i] not in tau_dict:
                    b_later = True
                else:
                    b_later = tau_dict[self.center[i]] > tau_dict[item]
                a_later = False
                dist += a_later != b_later
        
        p_dist += math.comb(len(remaining_tau), 2) * self.p
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
    
    def PRIM_POS(self, j: np.integer, item_count: int):
        return np.exp(-self.beta * j) / np.exp(-self.beta * np.arange(1, item_count + 1)).sum(-1)

    def PRIM_POS_SEQ(self, j: int, item_count: int, before: bool):
        if j == 0 and before:
            return 0

        prob_before = self.PRIM_POS(np.arange(1, j + 1), item_count).sum(0)
        if before:
            return prob_before
        return 1 - prob_before
    
    def choice_modeling_probs(self, ref_ranking: frozenset, k: int, n: int, null_val: int = 0) -> tuple[npt.NDArray[np.float64], npt.NDArray]:
        center_set = set(self.center[:k])
        A_null = ref_ranking.union({null_val})
        A_bar = [a for a in ref_ranking if a not in center_set]
        L = np.array([null_val] + A_bar + self.center[:k])
        
        r = len(A_bar)
        ell = len(self.center[:k])
        m = len(L)

        # Reversing
        cur_arr = [(self.center[i], r + 1 + i) for i in reversed(range(k))]
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
                    DP_table[:, jind, q] = (DP_table[:, jind, q-1] * self.PRIM_POS_SEQ(j, max_pos, before=False) 
                                            + DP_table[:, jind-1, q-1] * self.PRIM_POS_SEQ(j-1, max_pos, before=True))
                DP_table[:, k, q] = DP_table[:, k, q-1]
            else:
                #print("Second case")
                for j in range(1, k - ell + q + 1):
                    jind = j - 1
                    DP_table[:, jind, q] = DP_table[:, jind, q-1] * self.PRIM_POS_SEQ(j, max_pos, before=False)
                    DP_table[cur_ind, jind, q] = self.PRIM_POS(j, max_pos) * DP_table[:, jind:, q-1].sum((0, 1))
            #print("Loop for q:", q)
            #print(DP_table)
        
        #print("Total probs:", DP_table.sum((0, 1)))
        item_probs = DP_table[:, :k, ell].sum((1))
        return item_probs, L
