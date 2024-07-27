import argparse

import numpy as np
from math import log2, pow, log, sqrt

from tqdm import tqdm


def load_data(filename):
    global data
    file = open("../Data/22d_uniform.txt", 'r')
    data = []
    a = file.readlines()
    for i in a:
        d = i.strip().split()
        data.append((int(d[0]), int(d[1])))


class Tree2D:
    def __init__(self, n):
        self.n = n
        self.tree = np.array([[0] * (2 * self.n - 1) for _ in range(2 * self.n - 1)])

    def true_add(self, v1, v2):
        self.tree[v1 + self.n - 1][v2 + self.n - 1] += 1

    def true_build(self):
        for i in range(2 * self.n - 2, -1, -1):
            for j in range(2 * self.n - 2, -1, -1):
                if self.n - 1 <= j <= 2 * self.n - 1:
                    if i < self.n - 1:
                        self.tree[i][j] = self.tree[2 * i + 1][j] + self.tree[2 * i + 2][j]
                else:
                    self.tree[i][j] = self.tree[i][2 * j + 1] + self.tree[i][2 * j + 2]

    def add_noise_lap(self, eps):
        z = np.random.laplace(0, 1 / eps, size=(2 * self.n - 1, 2 * self.n - 1))
        self.tree = self.tree + z

    def add_noise_gaussian(self, eps, delta):
        sigma = sqrt(2 * log(2 / delta)) / eps
        z = np.random.normal(0, sigma, size=(2 * self.n - 1, 2 * self.n - 1))
        self.tree = self.tree + z

    def get_node(self, l, r):
        nodes = []
        start = l
        next = l
        base = int(log2(self.n)) + 1
        level = int(log2(self.n)) + 1
        index = l
        segment = []
        while next < r:
            if index % 2 == 0:
                save_next = next
                next += pow(2, base - level)
                if next < r:
                    level -= 1
                    index = index // 2
                else:
                    segment.append((start, save_next, level, index))
                    level = int(log2(self.n)) + 1
                    start = save_next + 1
                    index = start
                    next = start
            else:
                segment.append((start, next, level, index))
                level = int(log2(self.n)) + 1
                start = next + 1
                index = next + 1
                next += 1
        for (_, _, level, index) in segment:
            nodes.append(2 ** (level - 1) + index - 1)
            # print()
        return nodes

    # 2 dimensional range query
    def range_query(self, r1, l1, r2, l2):
        result = 0
        first_leyer = self.get_node(r1, l1)
        second_layer = self.get_node(r2, l2)
        for i in first_leyer:
            for j in second_layer:
                result += self.tree[int(i)][int(j)]
        return result


def print_info(file):
    file.write("epsilon:" + str(eps) + "\n")
    file.write("delta:" + str(delta) + "\n")
    # file.write("domain size:" + str(B) + "\n")

    file.write("pureDP Linf error:" + str(error1_5) + "\n")
    file.write("pureDP 50\% error:" + str(error1_1) + "\n")
    file.write("pureDP 90\% error:" + str(error1_2) + "\n")
    file.write("pureDP 95\% error:" + str(error1_3) + "\n")
    file.write("pureDP 99\% error:" + str(error1_4) + "\n")
    file.write("pureDP average error:" + str(error1_6) + "\n")

    file.write("approxDP Linf error:" + str(error2_5) + "\n")
    file.write("approxDP 50\% error:" + str(error2_1) + "\n")
    file.write("approxDP 90\% error:" + str(error2_2) + "\n")
    file.write("approxDP 95\% error:" + str(error2_3) + "\n")
    file.write("approxDP 99\% error:" + str(error2_4) + "\n")
    file.write("approxDP average error:" + str(error2_6) + "\n")


if __name__ == "__main__":
    global data
    parser = argparse.ArgumentParser(description='optimal small domain range counting for shuffle model')
    parser.add_argument('--epi', type=float, default=5, help='privacy budget')
    parser.add_argument('--rep', type=int)
    opt = parser.parse_args()
    load_data("1")

    # print(tree.tree)
    B = 16
    n = 1e7
    delta = 1 / (n * n)
    # eps = 10
    eps = opt.epi
    delta_s = delta / pow(log2(B)+1, 2)
    eps_s = eps / pow(log2(B)+1, 2)

    true = Tree2D(B)
    noise_pure = Tree2D(B)
    noise_approx = Tree2D(B)
    print("initialize")
    for v1, v2 in tqdm(data):
        noise_pure.true_add(v1, v2)
        noise_approx.true_add(v1, v2)
        true.true_add(v1,v2)
    noise_pure.true_build()
    noise_pure.add_noise_lap(eps_s)
    noise_approx.true_build()
    noise_approx.add_noise_gaussian(eps_s, delta_s)
    true.true_build()
    print(noise_pure.tree[0])
    print(noise_approx.tree[0])
    print(true.tree[0])
    error1 = []
    error2 = []
    for r1 in range(B):
        for l1 in range(r1+1, B):
            for r2 in range(B):
                for l2 in range(r2+1, B):
                    noise_pure_result = noise_pure.range_query(r1, l1, r2, l2)
                    noise_apporox_result = noise_approx.range_query(r1, l1, r2, l2)
                    true_result = true.range_query(r1, l1, r2, l2)
                    error1.append(abs(noise_pure_result - true_result))
                    error2.append(abs(noise_apporox_result - true_result))
    error1.sort()
    error1_1 = error1[int(len(error1) * 0.5)]
    error1_2 = error1[int(len(error1) * 0.9)]
    error1_3 = error1[int(len(error1) * 0.95)]
    error1_4 = error1[int(len(error1) * 0.99)]
    error1_5 = max(error1)
    error1_6 = np.average(error1)
    # print("pure", error1_1, error1_2, error1_3, error1_4, error1_5, error1_6)
    error2.sort()
    error2_1 = error2[int(len(error2) * 0.5)]
    error2_2 = error2[int(len(error2) * 0.9)]
    error2_3 = error2[int(len(error2) * 0.95)]
    error2_4 = error2[int(len(error2) * 0.99)]
    error2_5 = max(error2)
    error2_6 = np.average(error2)
    # print("approx", error2_1, error2_2, error2_3, error2_4, error2_5, error2_6)
    out_file = open("../log/Multi/central/" + str(1) + "_" + "eps=" + str(eps) + ".txt", 'w')
    print_info(out_file)
    out_file.close()
    print('finish')