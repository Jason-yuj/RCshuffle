import argparse
import time
import numpy as np
import math
import collections
from math import log, log2, ceil, floor, sqrt
from bisect import bisect_right, bisect_left

from tqdm import tqdm


def load_data(file_name):
    global data
    file = open(file_name, 'r')
    data = []
    a = file.readlines()
    for i in a:
        data.append(int(i))


def get_node(B, l, r):
    branch = 2
    nodes = []
    start = l
    next = l + 1
    base = int(math.log(B, branch))
    level = int(math.log(B, branch))
    index = l
    segment = []
    while next <= r:
        if index % branch == 0:
            save_next = next
            next += pow(branch, base - level + 1) - pow(branch, base - level)
            if next <= r:
                level -= 1
                index = index // branch
            else:
                segment.append((start, save_next, level, index))
                level = int(math.log(B, branch))
                start = save_next
                index = start
                next = start + 1
        else:
            segment.append((start, next, level, index))
            level = int(math.log(B, branch))
            start = next
            index = next
            next += 1
    for (i, j, level, index) in segment:
        nodes.append(int(((branch * pow(branch, level-1) - 1) / (branch - 1)) + index))
    return nodes


def pure_dp_noise():
    global B
    sensitivity = math.log2(B) + 1
    z = np.random.laplace(0, sensitivity / eps)
    return z


def approx_dp_noise():
    global B
    global eps
    global delta
    sensitivity = math.log2(B) + 1
    eps_s = eps / sensitivity
    delta_s = delta / sensitivity
    sigma = sqrt(2*log(2/delta_s))/eps_s
    z = np.random.normal(0, sigma)
    return z


def print_info(file):
    file.write("epsilon:" + str(eps) + "\n")
    file.write("delta:" + str(delta) + "\n")
    file.write("domain size:" + str(B) + "\n")

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='optimal small domain range counting for shuffle model')
    parser.add_argument('--dataset', type=str, default='uniform',
                        help='input data set')
    parser.add_argument('--epi', type=float, default=5, help='privacy budget')
    parser.add_argument('--rep', type=int)
    opt = parser.parse_args()
    global B
    global n
    global eps
    global delta
    global error1
    global error2
    global data
    n = 10000000
    B = pow(2, 30)
    eps = opt.epi
    delta = 1 / (n ** 2)
    error1 = []
    error2 = []
    in_file = opt.dataset

    # if in_file == "uniform":
    #     file_name = "./uniform.txt"
    # elif in_file == "AOL":
    #     file_name = "./AOL.txt"
    # elif in_file == "zipf":
    #     file_name = "./zipf.txt"
    # elif in_file == "gaussian":
    #     file_name = "./gaussian.txt"
    # elif in_file == "netflix":
    #     file_name = "./netflix.txt"
    # else:
    #     file_name = "./uniform.txt"
    load_data("../uniform.txt")

    if in_file == "AOL" or in_file == "netflix":
        distinct = set(data)
        domain = len(distinct)
        B = pow(2, math.ceil(math.log(domain) / math.log(2)))
        n = len(data)
    laplace_noise = {}
    gaussian_noise = {}
    for i in tqdm(range(100000)):
        l = np.random.randint(0, B)
        h = np.random.randint(0, B)
        while h == l:
            h = np.random.randint(0, B)
        nodes = get_node(B, min(l, h), max(l, h))

        pure_error = 0
        approx_error = 0
        for node in nodes:
            if node in laplace_noise.keys():
                pure_error += laplace_noise[node]
                approx_error += gaussian_noise[node]
            else:
                this_pure_noise = pure_dp_noise()
                pure_error += this_pure_noise
                laplace_noise[node] = this_pure_noise
                this_approx_noise = approx_dp_noise()
                approx_error += this_approx_noise
                gaussian_noise[node] = this_approx_noise
        error1.append(abs(pure_error))
        error2.append(abs(approx_error))
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
    out_file = open("../log/Large1D/central/" + str(opt.rep) + "_eps=" + str(eps) + ".txt", 'w')
    print_info(out_file)
    out_file.close()
    print('finish')