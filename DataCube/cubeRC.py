import numpy as np
from math import log2, pow, log, floor

from itertools import permutations, combinations
from tqdm import tqdm


def find_opt_cube(epi, d, L, attri, eps, delta, n):
    l = 0
    total = int(pow(2, d))
    delta_s = delta / 16
    eps_s = eps / 16
    mu_1 = 32 * log(2 / delta_s) / (eps_s * eps_s)
    sample_prob = mu_1 / n
    r = n * sample_prob * (1 - sample_prob)
    # r = 2 * pow(4, d) / pow(epi, 2)
    save = None
    while abs(r - l) > 1 / pow(epi, 2):
        cur = (l + r) / 2
        for s in range(1, total + 1):
            flag, R = feasible(L, cur, s, attri, eps, delta, n)
            if flag:
                r = cur
                save = R
                break
        if s == total:
            l = cur
        # print(save)
    return save


# by default, we will include the base cuboid
def feasible(L, theta, s, attri, eps, delta, n):
    temp = L.copy()

    R = {(0, 1, 2, 3)}
    cov = set()
    delta_s = delta / s
    eps_s = eps / s
    mu = 32 * log(2 / delta_s) / (eps_s * eps_s)
    sample_prob = mu / n
    var = n * sample_prob * (1 - sample_prob)
    # calculate bound
    bound = theta / var
    # coverage for base cuboid
    for cuboid in temp:
        mag = 1
        for a in (0, 1, 2,3 ):
            if a not in cuboid:
                mag = mag * attri[a]
        if mag <= bound:
            cov.add(cuboid)
    temp.remove((0, 1, 2, 3))
    # greedy
    for i in range(s - 1):
        save = {}
        cur_cuboid = None
        max_cov = 0
        for cuboid in temp:
            temp_cov = set()
            for pot in temp:
                mag = 1
                if set(pot).intersection(set(cuboid)) == set(pot):
                    for a in cuboid:
                        if a not in pot:
                            mag = mag * attri[a]
                    if mag <= bound:
                        temp_cov.add(pot)
            cov_count = len(temp_cov.difference(cov))
            if cov_count >= max_cov:
                max_cov = cov_count
                cur_cuboid = cuboid
                save[cuboid] = temp_cov
        cov = cov.union(save[cur_cuboid])
        R.add(cur_cuboid)
    if cov == L:
        print(R)
        return True, R
    else:
        return False, None


if __name__ == '__main__':
    n = 1e7
    eps = 5
    delta = 1 / (n * n)

    attri = [8, 8, 4, 4]
    L = {(0, 1, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (1,),
         (2,), (0,), (3,), ()}

    d = 4
    delta_s = delta / 4
    eps_s = eps / 4
    mu_1 = 32 * log(2 / delta_s) / (eps_s * eps_s)
    sample_prob = mu_1 / n
    test = n * sample_prob * (1 - sample_prob)
    find_opt_cube(eps, d, L, attri, eps, delta, n)
