import argparse
import collections
import math
from bisect import bisect_right, bisect_left

import numpy as np
from tqdm import tqdm


def load_data(filename):
    global data
    file = open(filename, 'r')
    data = []
    a = file.readlines()
    for i in a:
        data.append(int(i))


def pre_process():
    global true_frequency
    true_frequency = np.zeros(B)
    true_counter = collections.Counter(data)
    # print(counter)
    # true_frequency = np.ones(B)
    for i in range(0, B):
        if i in true_counter.keys():
            true_frequency[i] += true_counter[i]


def local_randomizer(x, p):
    global number_msg
    global messages
    messages.append(B + x - 1)
    noise_msg_1 = np.random.binomial(1, p, size=2 * B - 1)
    noise_msg_1 = np.where(noise_msg_1 == 1)[0]
    noise_msg_2 = noise_msg_1[noise_msg_1 != 0]
    noise_msg_2 = ((noise_msg_2 + 1) // 2) - 1
    messages += noise_msg_1.tolist()
    messages += noise_msg_2.tolist()
    return


def analyzer():
    global opt_frequency
    global messages
    opt_frequency = np.zeros(2 * B - 1)
    fe_counter = collections.Counter(messages)
    for i in range(0, 2 * B - 1):
        if i in fe_counter.keys():
            opt_frequency[i] += fe_counter[i]
    for i in range(B - 1, 0, -1):
        t = pow(-1, math.floor(math.log2(max(1, i))) + (math.log2(B) % 2))
        # t = 0
        opt_frequency[i - 1] = t * (opt_frequency[(i - 1)]) + (opt_frequency[2 * i - 1] + opt_frequency[2 * i])
    for i in range(1, 2 * B):
        t = pow(-1, math.floor(math.log2(max(1, i))) + (math.log2(B) % 2))
        opt_frequency[i - 1] -= t * mu_1
    return


def get_node(B, l, r):
    nodes = []
    start = l
    next = l
    base = int(math.log2(B)) + 1
    level = int(math.log2(B)) + 1
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
                level = int(math.log2(B)) + 1
                start = save_next + 1
                index = start
                next = start
        else:
            segment.append((start, next, level, index))
            level = int(math.log2(B)) + 1
            start = next + 1
            index = next + 1
            next += 1
    for (_, _, level, index) in segment:
        nodes.append(2 ** (level - 1) + index - 1)
        # print()
    # print(segment)
    return nodes


def range_query(l, h):
    global opt_frequency
    nodes = get_node(B, min(l, h), max(l, h))
    # print(nodes)
    result = 0
    for node in nodes:
        result += opt_frequency[node]
    return result


def true_result(l, h):
    global data
    left = bisect_left(data, min(l, h))
    right = bisect_left(data, max(l, h))
    return right - left


def print_info(file):
    file.write("epsilon:" + str(eps) + "\n")
    file.write("delta:" + str(delta) + "\n")
    file.write("number of participants:" + str(n) + "\n")
    file.write("domain size:" + str(B) + "\n")
    file.write("mu:" + str(mu_1) + "\n")
    file.write("dataset:" + "uniform" + "\n")

    file.write("expected number of message / user:" + str(expected_msg) + "\n")
    # file.write("read number of message :" + str(number_msg) + "\n")
    file.write("real number of message / user:" + str(len(messages) / n) + "\n")

    file.write("Linf error:" + str(error_5) + "\n")
    file.write("50\% error:" + str(error_1) + "\n")
    file.write("90\% error:" + str(error_2) + "\n")
    file.write("95\% error:" + str(error_3) + "\n")
    file.write("99\% error:" + str(error_4) + "\n")
    file.write("average error:" + str(error_6) + "\n")


if __name__ == '__main__':
    global delta
    global eps
    global data
    global B
    global true_frequency
    global opt_frequency
    global n
    global mu_1
    global test
    parser = argparse.ArgumentParser(description='optimal small domain range counting for shuffle model')
    parser.add_argument('--n', type=int, help='total number of user')
    parser.add_argument('--B', '--b', type=int, help='domain range, B << n')
    parser.add_argument('--dataset', type=str, default='uniform',
                        help='input data set')
    parser.add_argument('--epi', type=float, default=5, help='privacy budget')
    parser.add_argument('--rep', type=int)
    opt = parser.parse_args()
    # test = 0.0
    B = opt.B
    n = opt.n
    delta = 1 / (n * n)
    eps = opt.epi

    number_msg = 0
    messages = []
    print("preprocess")
    in_file = opt.dataset

    if in_file == "uniform":
        file_name = "./uniform.txt"
    elif in_file == "AOL":
        file_name = "./AOL.txt"
    elif in_file == "zipf":
        file_name = "./zipf.txt"
    elif in_file == "gaussian":
        file_name = "./gaussian.txt"
    elif in_file == "netflix":
        file_name = "./netflix.txt"
    else:
        file_name = "./uniform.txt"
    load_data(file_name)

    if in_file == "AOL" or in_file == "netflix":
        distinct = set(data)
        domain = len(distinct)
        B = pow(2, math.ceil(math.log(domain) / math.log(2)))
        n = len(data)
    else:
        B = opt.B
    delta_s = delta / (math.log2(B) + 1)
    eps_s = eps / (math.log2(B) + 1)
    mu_1 = 32 * math.log(2 / delta_s) / (eps_s * eps_s)
    # mu_1 = mu_list[(n, eps)]
    print(mu_1)
    pre_process()
    sample_prob = mu_1 / n
    # sample_prob = 0.01
    # mu_1 = sample_prob * n
    print(sample_prob)
    print("initialize")
    for j in tqdm(range(n)):
        local_randomizer(data[j], sample_prob)
    # print(messages)
    analyzer()
    # print(opt_frequency)
    expected_msg = 1 + sample_prob * (4 * B - 3)
    # print(expected_msg)
    error = []
    data.sort()
    for l in tqdm(range(B)):
        for h in range(l + 1, B):
            noise_result = range_query(l, h)
            true = true_result(l, h)
            # print(l,h,noise_result,true)
            # print(l, h, noise_result, true)
            error.append(abs(noise_result - true))
            # if abs(noise_result - true) != 0:
            #     print(abs(noise_result - true))
    # print(opt_frequency)
    # print(error)
    global error_1
    global error_2
    global error_3
    global error_4
    global error_5
    global error_6
    error.sort()
    error_1 = error[int(len(error) * 0.5)]
    error_2 = error[int(len(error) * 0.9)]
    error_3 = error[int(len(error) * 0.95)]
    error_4 = error[int(len(error) * 0.99)]
    error_5 = max(error)
    error_6 = np.average(error)
    out_file = open("./log/Small1D/OPTs/" + str(opt.rep) + "_" + str(opt.dataset) + "_B=" + str(B) + "_n=" + str(n) + "_eps=" + str(eps) + ".txt",
                    'w')
    print_info(out_file)
    print("finish")
    out_file.close()
