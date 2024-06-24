import numpy as np
from math import log2, pow, log, floor

from tqdm import tqdm


def load_data(filename):
    global data
    file = open("../Data/2d_uniform.txt", 'r')
    data = []
    a = file.readlines()
    for i in a:
        d = i.strip().split()
        data.append((int(d[0]), int(d[1])))


class Tree2D:
    def __init__(self, n, mu):
        self.mu = mu
        self.n = n
        self.tree = [[0] * (2 * self.n - 1) for _ in range(2 * self.n - 1)]

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

    def add(self, v1, v2, p):
        # true msg
        msg = 1
        self.tree[v1 + self.n - 1][v2 + self.n - 1] += 1
        # noise msg
        noise_msg_1 = np.random.binomial(1, p, size=(2 * self.n - 1, 2 * self.n - 1))
        noise_msg_1 = np.argwhere(noise_msg_1 == 1)
        for i, j in noise_msg_1:
            # all leaf node in aux tree
            if self.n - 1 <= j <= 2 * self.n - 1:
                if i != 0:
                    self.tree[((i + 1) // 2) - 1][j] += 1
                    msg += 1
            self.tree[i][j] += 1
            msg += 1
            # parent node in aux tree
            if j != 0:
                self.tree[i][((j + 1) // 2) - 1] += 1
                msg += 1
        return msg

    # for build the whole tree in a bottom-up manner
    def build(self):
        for i in range(2 * self.n - 2, -1, -1):
            for j in range(2 * self.n - 2, -1, -1):
                t_1 = pow(-1, floor(log2(max(1, j+1))) + (log2(self.n) % 2))
                t_2 = pow(-1, floor(log2(max(1, i+1))) + (log2(self.n) % 2))
                t = t_1*t_2
                # print(i, j, t)
                if self.n - 1 <= j <= 2 * self.n - 1:
                    if i < self.n - 1:
                        # self.tree[i-1][j-1] = t * self.tree[i-1][j-1] + self.tree[2 * i - 1][j-1] + self.tree[2 * i][j-1]
                        self.tree[i][j] = t * self.tree[i][j] + self.tree[2 * i + 1][j] + self.tree[2 * i + 2][j]
                else:
                    # self.tree[i-1][j-1] = t * self.tree[i-1][j-1] + self.tree[i-1][2 * j - 1] + self.tree[i-1][2 * j]
                    self.tree[i][j] = t * self.tree[i][j] + self.tree[i][2 * j + 1] + self.tree[i][2 * j + 2]
        # debias
        for i in range(2 * self.n - 2, -1, -1):
            for j in range(2 * self.n - 2, -1, -1):
                t_1 = pow(-1, floor(log2(max(1, j + 1))) + (log2(self.n) % 2))
                t_2 = pow(-1, floor(log2(max(1, i + 1))) + (log2(self.n) % 2))
                t = t_1 * t_2
                # print(i, j, t)
                self.tree[i][j] -= t * self.mu

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


if __name__ == "__main__":
    global data
    load_data("1")

    # print(tree.tree)
    B = 32
    n = 1e7
    delta = 1 / (n * n)
    eps = 10
    delta_s = delta / pow(log2(B)+1, 2)
    eps_s = eps / pow(log2(B)+1, 2)
    mu_1 = 32 * log(2 / delta_s) / (eps_s * eps_s)
    sample_prob = mu_1 / n
    print(sample_prob, mu_1)
    tree = Tree2D(B, mu_1)
    true = Tree2D(B, 1)
    print("initialize")
    t = 0
    # tree.add(1, 1, 0.5)
    for v1, v2 in tqdm(data):
        msg = tree.add(v1, v2, sample_prob)
        true.true_add(v1,v2)
        t += msg
    print(t)
    tree.build()
    true.true_build()
    print(tree.tree[1])
    print(true.tree[1])
    error = []
    for r1 in range(B):
        for l1 in range(r1+1, B):
            for r2 in range(B):
                for l2 in range(r2+1, B):
                    noise_result = tree.range_query(r1, l1, r2, l2)
                    true_result = true.range_query(r1, l1, r2, l2)
                    # print(noise_result, true_result)
                    error.append(abs(noise_result - true_result))
    error.sort()
    error_1 = error[int(len(error) * 0.5)]
    error_2 = error[int(len(error) * 0.9)]
    error_3 = error[int(len(error) * 0.95)]
    error_4 = error[int(len(error) * 0.99)]
    error_5 = max(error)
    error_6 = np.average(error)
    print(error_1, error_2, error_3, error_4, error_5, error_6)
    print(t / n)