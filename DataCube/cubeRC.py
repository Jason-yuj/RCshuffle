import numpy as np
from math import log2, pow, log, floor

import collections
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
        for a in (0, 1, 2, 3):
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
        # print(R)
        return True, R
    else:
        return False, None


def load_data(filename):
    global data
    file = open("../Data/cube_sample.txt", 'r')
    data = []
    a = file.readlines()
    for i in tqdm(a):
        d = i.strip().split()
        data.append((int(d[0]), int(d[1]), int(d[2]), int(d[3])))


def pre_process():
    global L_pre_level
    global L_level
    global L_pre
    global List_pre
    global child
    L_pre_level = {}
    L_level = {}
    child = {}
    # level for L_pre
    for cuboid in L_pre:
        if len(cuboid) in L_pre_level.keys():
            L_pre_level[len(cuboid)].append(cuboid)
        else:
            L_pre_level[len(cuboid)] = [cuboid]
    # level for L
    for cuboid in L:
        if len(cuboid) in L_level.keys():
            L_level[len(cuboid)].append(cuboid)
        else:
            L_level[len(cuboid)] = [cuboid]

    for cuboid in List_pre:
        if len(cuboid) + 1 in L_level.keys():
            save = None
            min_mag = 1000
            for From in L_level[len(cuboid) + 1]:

                if set(From).intersection(set(cuboid)) == set(cuboid):
                    mag = child[From][1]
                    for a in From:
                        if a not in cuboid:
                            mag = mag * attri[a]
                    if mag < min_mag:
                        save = From
                        min_mag = mag
            if cuboid in L_pre:
                child[cuboid] = (save, 1)
            else:
                child[cuboid] = (save, min_mag)
        else:
            child[cuboid] = (None, 1)
    # for i in child:
    #     print(i, child[i])


def local_randomizer(x, p, cells):
    global tree
    global L_pre_level
    global child
    noise_msg = np.random.binomial(1, p, size=len(cells))
    noise_msg_index = np.argwhere(noise_msg == 1)
    msg = [x]
    # for i in noise_msg_index:
    #     cell = cells[i[0]]
    #     msg.append(cell)
    #     cub = [i[0] for i in np.argwhere(np.array(cell) != -1)]
    #     cub_level = len(cub)
    #     if (cub_level - 1) in L_pre_level.keys():
    #         for cuboid in L_pre_level[cub_level - 1]:
    #             parent = [-1, -1, -1, -1]
    #             if child[cuboid][0] == tuple(cub):
    #                 for j in cuboid:
    #                     parent[j] = cell[j]
    #                 msg.append(tuple(parent))
    return msg


def analyzer():
    global L_pre_level
    global L_level
    global L_pre
    global List_pre
    global child
    global messages
    global attri
    global tree
    fe_counter = collections.Counter(messages)
    for key in fe_counter.keys():
        if key in tree.keys():
            tree[key] = fe_counter[key]
    keys = list(tree.keys())
    keys.sort(key=lambda x: list(x).count(-1))
    # print(keys)
    for key in keys:
        # not a base cell
        if -1 in key:
            cub = [i[0] for i in np.argwhere(np.array(key) != -1)]
            From = child[tuple(cub)][0]
            dim = list(set(From).difference(set(cub)))[0]
            total = tree[key]
            for i in range(attri[dim]):
                parent = list(key)
                parent[dim] = i
                total += tree[tuple(parent)]
            tree[key] = total
    # for i in tree:
    #     print(i, tree[i])


if __name__ == '__main__':
    global tree
    global L_pre_level
    global L_level
    global L_pre
    global List_pre
    global child
    global attri
    n = 1e7
    eps = 5
    delta = 1 / (n * n)
    attri = [8, 8, 4, 4]
    L = {(0, 1, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (1,),
         (2,), (0,), (3,), ()}
    List_pre = [(0, 1, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3),
                (2, 3), (1,),
                (2,), (0,), (3,), ()]
    d = 4
    # test = n * sample_prob * (1 - sample_prob)

    L_pre = find_opt_cube(eps, d, L, attri, eps, delta, n)
    pre_process()
    print(L_pre)
    delta_s = delta / len(L_pre)
    eps_s = eps / len(L_pre)
    mu_1 = 32 * log(2 / delta_s) / (eps_s * eps_s)
    sample_prob = mu_1 / n
    # pre_cube to all counters(cells)
    tree = {}
    cells = []
    for c in L_pre:
        atr_list = [[-1], [-1], [-1], [-1]]
        for a in c:
            atr_list[a] = [i for i in range(attri[a])]
        for x in atr_list[0]:
            for y in atr_list[1]:
                for m in atr_list[2]:
                    for n in atr_list[3]:
                        tree[(x, y, m, n)] = 0
                        cells.append((x, y, m, n))
    global data
    global t
    global messages
    t = 0
    print("preprocess")
    load_data("filename")
    print("initialize")
    messages = []
    for dt in tqdm(data):
        msg_l = local_randomizer(dt, sample_prob, cells)
        messages += msg_l
    analyzer()
    print(len(tree))
