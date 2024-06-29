import numpy as np
from math import log2, pow, log, floor, sqrt

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
    file = open("../Data/cube.txt", 'r')
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


def get_counter(x):
    global counter
    counter.append(x)
    cur_level = len([i[0] for i in np.argwhere(np.array(x) != -1)])
    for cuboid in L_pre:
        cub_level = len(cuboid)
        if cub_level < cur_level and cub_level in L_pre_level.keys():
            parent = [-1, -1, -1, -1]
            for j in cuboid:
                parent[j] = x[j]
            counter.append(tuple(parent))


def pure_dp(cells):
    global pure_dp_cube
    global L_pre
    global counter
    pure_dp_cube = collections.Counter(counter)
    z = np.random.laplace(0, len(L_pre) / eps, size=len(cells))
    for i in range(len(cells)):
        # print(cells[i], pure_dp_cube[cells[i]])
        pure_dp_cube[cells[i]] = pure_dp_cube[cells[i]] + z[i]


def approx_dp(cells):
    global approx_dp_cube
    global L_pre
    global counter
    global eps
    global delta
    eps_s = eps / len(L_pre)
    delta_s = delta / len(L_pre)
    sigma = sqrt(2 * log(2 / delta_s)) / eps_s
    approx_dp_cube = collections.Counter(counter)
    z = np.random.normal(0, sigma, size=len(cells))
    for i in range(len(cells)):
        approx_dp_cube[cells[i]] = approx_dp_cube[cells[i]] + z[i]


def post_dataCube_true():
    global data
    global List_pre
    global ture_frequency
    global all_cells
    ture_frequency = collections.Counter(data)
    all_cells = []
    atr_list = [[-1], [-1], [-1], [-1]]
    for a in (0,1,2,3):
        atr_list[a] = [i for i in range(-1, attri[a], 1)]
    for x in atr_list[0]:
        for y in atr_list[1]:
            for m in atr_list[2]:
                for n in atr_list[3]:
                    all_cells.append((x, y, m, n))
    all_cells.sort(key=lambda x: list(x).count(-1))
    for cell in all_cells:
        # not a base cell
        if -1 in cell:
            cub = [i[0] for i in np.argwhere(np.array(cell) != -1)]
            From = child[tuple(cub)][0]
            dim = list(set(From).difference(set(cub)))[0]
            total = ture_frequency[cell]
            for i in range(attri[dim]):
                parent = list(cell)
                parent[dim] = i
                total += ture_frequency[tuple(parent)]
            ture_frequency[cell] = total


def post_dataCube(cube):
    global List_pre
    global all_cells
    all_cells = []
    atr_list = [[-1], [-1], [-1], [-1]]
    noise_cube = {}
    for a in (0,1,2,3):
        atr_list[a] = [i for i in range(-1, attri[a], 1)]
    for x in atr_list[0]:
        for y in atr_list[1]:
            for l in atr_list[2]:
                for m in atr_list[3]:
                    all_cells.append((x, y, l, m))
    all_cells.sort(key=lambda x: list(x).count(-1))
    for cell in all_cells:
        # not a base cell
        if -1 in cell and cell not in cube.keys():
            cub = [i[0] for i in np.argwhere(np.array(cell) != -1)]
            From = child[tuple(cub)][0]
            dim = list(set(From).difference(set(cub)))[0]
            total = 0
            for i in range(attri[dim]):
                parent = list(cell)
                parent[dim] = i
                total += noise_cube[tuple(parent)]
            noise_cube[cell] = total
        else:
            noise_cube[cell] = cube[cell]
    # for i in fe_counter:
    #     if fe_counter[i] not in [1,2,4,8,16,32,64,128,256, 1024]:
    #         print(i, fe_counter[i])
    return noise_cube


if __name__ == '__main__':
    global pure_dp_cube
    global L_pre_level
    global L_level
    global L_pre
    global List_pre
    global child
    global attri
    global d
    global counter
    n = 1e7
    eps = 10
    delta = 1 / (n * n)
    attri = [8, 8, 4, 4]
    L = {(0, 1, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (1,),
         (2,), (0,), (3,), ()}
    List_pre = [(0, 1, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3),
                (2, 3), (1,),
                (2,), (0,), (3,), ()]
    d = 4
    L_pre = find_opt_cube(eps, d, L, attri, eps, delta, n)
    pre_process()
    print(L_pre)
    delta_s = delta / len(L_pre)
    eps_s = eps / len(L_pre)
    # pre_cube to all counters(cells)
    cells = []
    for c in L_pre:
        atr_list = [[-1], [-1], [-1], [-1]]
        for a in c:
            atr_list[a] = [i for i in range(attri[a])]
        for x in atr_list[0]:
            for y in atr_list[1]:
                for l in atr_list[2]:
                    for m in atr_list[3]:
                        cells.append((x, y, l, m))
    global data
    global error1
    global error2
    global all_cells
    global approx_dp_cube
    global pure_dp_cube
    global ture_frequency
    error = []
    counter = []
    print("preprocess")
    load_data("filename")
    for dt in tqdm(data):
        get_counter(dt)
    approx_dp(cells)
    pure_dp(cells)
    print("initialize")
    print("finish")
    post_dataCube_true()
    pure_dp_result = post_dataCube(pure_dp_cube)
    approx_dp_result = post_dataCube(approx_dp_cube)
    # for i in pure_dp_result:
    #     print(i, pure_dp_result[i], approx_dp_result[i], ture_frequency[i])
    error1 = []
    error2 = []
    for i in all_cells:
        error1.append(abs(pure_dp_result[i] - ture_frequency[i]))
        error2.append(abs(approx_dp_result[i] - ture_frequency[i]))
    error1.sort()
    error1_1 = error1[int(len(error1) * 0.5)]
    error1_2 = error1[int(len(error1) * 0.9)]
    error1_3 = error1[int(len(error1) * 0.95)]
    error1_4 = error1[int(len(error1) * 0.99)]
    error1_5 = max(error1)
    error1_6 = np.average(error1)
    print("pure", error1_1, error1_2, error1_3, error1_4, error1_5, error1_6)
    error2.sort()
    error2_1 = error2[int(len(error2) * 0.5)]
    error2_2 = error2[int(len(error2) * 0.9)]
    error2_3 = error2[int(len(error2) * 0.95)]
    error2_4 = error2[int(len(error2) * 0.99)]
    error2_5 = max(error2)
    error2_6 = np.average(error2)
    print("approx", error2_1, error2_2, error2_3, error2_4, error2_5, error2_6)
