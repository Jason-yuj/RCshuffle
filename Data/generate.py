import argparse
import bisect
import math
import random

import numpy as np


# generate uniform data
def generate_uniform(n, B):
    file = open("uniform.txt", 'w')
    # file.write(str(n)+'\n')
    # file.write(str(B)+'\n')
    for _ in range(n):
        i = np.random.randint(0, B)
        file.write(str(i)+'\n')
    print("finish")
    file.close()


class Zipfian:
    def __init__(self, alpha):
        self.alpha = alpha
        self.maxnoise = 1024

        self.accprob = []
        cumprob, prob = 0, 0

        for i in range(1, self.maxnoise):
            prob = 1.0 / pow(i, alpha)
            cumprob += prob
            self.accprob.append(cumprob)

        self.largeprob = self.accprob[-1]
        # print(self.accprob)

    def Generate(self):
        randness = random.uniform(0, self.largeprob)
        randnum = bisect.bisect_left(self.accprob, randness) + 1
        return randnum


def gen_zipf(n, B):
    alpha = 1.5
    zipf = Zipfian(alpha)
    zipflist = [[zipf.Generate()] for i in range(n)]
    # print (zipflist)
    file = open("zipf.txt", "w")
    for line in zipflist:
        file.write(str(line[0]) + '\n')
    print("finish")
    file.close()


def gen_gaussian(n, B):
    mean = B/3
    std = 20
    data = np.random.normal(mean, std, size=n)
    # t = data > B
    # print(np.where(data > B))
    file = open("gaussian.txt", "w")
    for i in data:
        file.write(str(int(i)) + '\n')
    print("finish")
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='optimal small domain range counting for shuffle model')
    parser.add_argument('--n', type=int, help='total number of user')
    parser.add_argument('--B', '--b', type=int, help='domain range')
    parser.add_argument('--dataset', type=str, default='uniform',
                        help='input data set')
    opt = parser.parse_args()
    n = opt.n
    b = opt.B
    t = opt.dataset
    if t == "uniform":
        generate_uniform(n, b)
    elif t == "gaussian":
        gen_gaussian(n, b)
    else:
        gen_zipf(n, b)
