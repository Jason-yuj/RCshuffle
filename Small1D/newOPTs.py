import argparse
import collections
import math
from bisect import bisect_right, bisect_left

import multiprocessing
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
    global B
    global size
    size = int((branch * B - 1) / (branch - 1))


def sub_process(i, data, p):
    global messages
    np.random.seed(i)
    msg = []
    for d in tqdm(data):
        local_msg = local_randomizer(d, p)
        msg.extend(local_msg)
    messages[i] = msg


def local_randomizer(x, p):
    global number_msg
    global messages
    global branch
    global size
    local_messages = [size - B + x]
    noise_msg_1 = np.random.binomial(1, p, size=size)
    noise_msg_1 = np.where(noise_msg_1 == 1)[0]
    noise_msg_2 = noise_msg_1[noise_msg_1 != 0]
    noise_msg_2 = (noise_msg_2 // branch - (noise_msg_2 % branch == 0))
    noise_msg_2 = noise_msg_2[noise_msg_2 >= 0]
    local_messages.extend(noise_msg_1.tolist())
    local_messages.extend(noise_msg_2.tolist())
    return local_messages


def analyzer():
    global opt_frequency
    global messages
    global total_msg
    opt_frequency = np.zeros(size)
    total_msg = []
    for i in messages.values():
        # print(len(i))
        total_msg.extend(i)
    opt_frequency = np.zeros(size)
    fe_counter = collections.Counter(total_msg)
    for i in range(0, size):
        if i in fe_counter.keys():
            opt_frequency[i] += fe_counter[i]
    for i in range(size - B, 0, -1):
        level = 0
        cur = 0.0
        while cur < i:
            cur += pow(branch, level)
            level += 1
        # print(i, level - 1)
        t = pow(-1, level - 1 + (math.log(B, branch) % 2))
        # print(i, t)
        opt_frequency[i - 1] = t * (opt_frequency[(i - 1)])
        for j in range(0, branch):
            opt_frequency[i - 1] += opt_frequency[branch * (i - 1) + j + 1]
    for i in range(1, size+1):
        level = 0
        cur = 0.0
        while cur < i:
            cur += pow(branch, level)
            level += 1
        t = pow(-1, level - 1 + (math.log(B, branch) % 2))
        opt_frequency[i - 1] -= t * mu_1
    return


def get_node(B, l, r):
    global branch
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


def checker(l, h):
    global data
    res = 0
    for i in data:
        if l <= i < h:
            res += 1
    return res


def range_query(l, h):
    global opt_frequency
    nodes = get_node(B, min(l, h), max(l, h))
    # print(nodes)
    result = 0
    # print(l, h, nodes)
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
    file.write("real number of message / user:" + str(len(total_msg) / n) + "\n")

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
    global domain
    global size
    global messages
    global total_msg
    multiprocessing.set_start_method("fork")
    parser = argparse.ArgumentParser(description='optimal small domain range counting for shuffle model')
    parser.add_argument('--n', type=int, help='total number of user')
    parser.add_argument('--B', '--b', type=int, help='domain range, B << n')
    parser.add_argument('--dataset', type=str, default='uniform',
                        help='input data set')
    parser.add_argument('--epi', type=float, default=5, help='privacy budget')
    parser.add_argument('--rep', type=int)
    opt = parser.parse_args()
    # test = 0.0
    branch = 2
    B = opt.B
    # B = 1024
    n = opt.n
    # n = 10000000
    delta = 1 / (n * n)
    eps = opt.epi
    # eps = 5

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
        B = pow(branch, math.ceil(math.log(domain) / math.log(branch)))
        n = len(data)
    else:
        domain = opt.B
        B = opt.B
        n = len(data)
    # distinct = set(data)
    # domain = len(distinct)
    # B = pow(branch, math.ceil(math.log(domain) / math.log(branch)))
    # n = len(data)
    delta_s = delta / (math.log(B, branch) + 1)
    eps_s = eps / (math.log(B, branch) + 1)
    mu_1 = 32 * math.log(2 / delta_s) / (eps_s * eps_s)
    # mu_1 = mu_list[(n, eps)]
    print(mu_1)
    pre_process()
    sample_prob = mu_1 / n
    # sample_prob = 0.01
    # mu_1 = sample_prob * n
    print(sample_prob)
    print("initialize")
    process_num = 5
    index = n // 5
    result = []
    manager = multiprocessing.Manager()
    messages = manager.dict()
    for i in range(process_num):
        # Try to make  parameters locally
        if i < process_num - 1:
            left = index * i
            right = index * (i + 1)
        else:
            left = index * i
            right = n
        # print(i, left, right)
        messages[i] = []
        local_data = data[left:right]
        result.append(multiprocessing.Process(target=sub_process, args=(i, local_data, sample_prob)))
        result[i].start()
    for i in range(process_num):
        result[i].join()
    # print(messages)
    analyzer()
    # print(opt_frequency)
    expected_msg = 1 + (2 * size - 1) * sample_prob
    # print(expected_msg)
    error = []
    data.sort()
    for l in tqdm(range(domain)):
        for h in range(l + 1, domain):
            noise_result = range_query(l, h)
            true = true_result(l, h)
            error.append(abs(noise_result - true))
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
    out_file = open("./log/Small1D/newOPTs/" + str(opt.rep) + str(opt.dataset) + "_B=" + str(B) + "_n=" + str(n) + "_eps=" + str(eps) + "_2.txt",
                    'w')
    print_info(out_file)
    print("finish")
    out_file.close()
