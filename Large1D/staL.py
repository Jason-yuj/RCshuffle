import argparse
import collections
import os
import time
from math import log, log2, ceil, floor
from bisect import bisect_right, bisect_left

import sys
import numpy as np
from tqdm import tqdm
import multiprocessing


def load_data(filename):
    global data
    file = open(filename, 'r')
    data = []
    a = file.readlines()
    for i in a:
        data.append(int(i))


def pre_process():
    global true_frequency
    global data
    global B
    global levelq
    global b
    global bertrand_primes
    global s
    global t
    # true_counter = collections.Counter(data)
    # print(counter)
    true_frequency = np.zeros(B)
    # for i in tqdm(range(0, B)):
    #     if i in true_counter.keys():
    #         true_frequency[i] += true_counter[i]
    levelq = []
    for i in range(int(log2(B)) + 1):
        if pow(2, i) < b:
            levelq.append(0)
        else:

            levelq.append(bertrand_primes[bisect_right(bertrand_primes, pow(2, i))])
            # print(pow(2, i), levelq[i], pow(2, i+1))
    for i in range(s, int(t) + 1):
        messages[i] = []


def sub_process_randomizer(i, local_data):
    global messages
    global s
    global t
    global msg_num
    np.random.seed()
    msg = {}
    # int(t) + 1
    for i in range(s, int(t) + 1):
        msg[i] = []
    for d in tqdm(local_data):
        for l in range(s, int(t) + 1):
            local_msg = local_randomizer(d, l)
            msg[l].extend(local_msg)
    for i in range(s, int(t) + 1):
        # if i < 3:
        #     print(len(msg[i]))
        messages[i] = msg
        # msg_num.value += len(msg[i])


def local_randomizer(x, l):
    global b
    global t
    global mu
    global levelq
    local_msg = []
    domain_size = pow(2, l)
    x_p = floor(x / pow(2, t - l))
    if domain_size < b:
        sample_prob = mu / n
        local_msg.append(x_p)
        y = np.random.binomial(1, sample_prob, size=domain_size)
        noise_msg = np.where(y)[0]
        local_msg.extend(noise_msg.tolist())
    else:
        q = levelq[l]
        u = np.random.randint(0, q - 1)
        v = np.random.randint(0, q)
        w = ((u * x_p + v) % q) % b
        local_msg.append((u, v, w))
        rou = mu * b / n
        fixed_msg = floor(rou)
        remaining_msg = rou - fixed_msg
        send_msg = fixed_msg + np.random.binomial(1, remaining_msg)
        for i in range(send_msg):
            u = np.random.randint(0, q - 1)
            v = np.random.randint(0, q)
            w = np.random.randint(0, b)
            local_msg.append((u, v, w))
    return local_msg


# def analyzer(l):
#     pass
def get_node(B, l, r):
    branch = 2
    nodes = []
    start = l
    next = l + 1
    base = int(log(B, branch))
    level = int(log(B, branch))
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
                level = int(log(B, branch))
                start = save_next
                index = start
                next = start + 1
        else:
            segment.append((start, next, level, index))
            level = int(log(B, branch))
            start = next
            index = next
            next += 1
    for (i, j, level, index) in segment:
        nodes.append(int(((branch * pow(branch, level-1) - 1) / (branch - 1)) + index))
    return nodes


def quick_power(x, y, mod):
    res = 1
    while y:
        if y & 1:
            res = res * x % mod
        x = x * x % mod
        y >>= 1
    return res


def counter_all(level):
    global messages
    global levelq
    global b
    domain_size = int(pow(2, level))
    i_frequency = np.zeros(domain_size)
    if domain_size < b:
        i_counter = collections.Counter(messages[level])
        for i in range(0, domain_size):
            if i in i_counter.keys():
                i_frequency[i] += i_counter[i]
    else:
        q = levelq[level]
        for m in messages[level]:
            u = m[0]
            v = m[1]
            w = m[2]
            invu = quick_power(u, q - 2, q)
            start_id = invu * (w - v + q) % q
            adding = invu * b % q
            id = start_id

            for j in range(0, ceil((q - 1 - w) / b), 1):
                if 0 <= id < domain_size:
                    i_frequency[id] += 1
                id += adding
                if id >= q:
                    id -= q
    return i_frequency.tolist()


def analyzer():
    global s
    global t
    global fe
    for level in range(int(s), 10):
        domain_size = pow(2, level)
        if domain_size < b:
            for i in range(domain_size):
                loc = (int(((2 * pow(2, level - 1) - 1) / (2 - 1)) + i))
                counter = total_msg[level].count(i)
                # print(len(messages[level]), counter)
                counter -= mu
                # print(counter, mu)
                fe[loc] = counter
        else:
            q = levelq[level]
            i_fe = counter_all(level)
            for i in range(domain_size):
                counter = i_fe[i]
                loc = (int(((2 * pow(2, level - 1) - 1) / (2 - 1)) + i))
                rou = mu * b / n
                collision_prob = 1.0 * (q / b) * (q % b + q - b) / (1.0 * q * (q - 1))
                res = (counter - n * rou / b - n * collision_prob) / (1 - collision_prob)
                fe[loc] = res


def sub_process_query(i):
    global error
    np.random.seed()
    local_error = []
    for _ in tqdm(range(0, 1000)):
        l = np.random.randint(0, B)
        h = np.random.randint(0, B)
        while h == l:
            h = np.random.randint(0, B)
        noise_result = range_query(l, h)
        true = true_result(l, h)
        # print(noise_result, true)
        local_error.append(abs(noise_result-true))
    error.extend(local_error)


def range_query(l, h):
    global b
    global n
    global mu
    global total_msg
    global fe
    res = 0
    nodes = get_node(B, min(l, h), max(l, h))
    for node in nodes:
        res += fe[node]
    return res


def true_result(l, h):
    global data
    left = bisect_left(data, min(l, h))
    right = bisect_left(data, max(l, h))
    return right - left


def print_info(file):
    file.write("epsilon:" + str(eps) + "\n")
    file.write("delta:" + str(delta) + "\n")
    file.write("number of participants:" + str(n) + "\n")
    file.write("large domain size:" + str(B) + "\n")
    file.write("reduced small domain:" + str(next) + "\n")
    # file.write("mu:" + str(mu_1) + "\n")

    # file.write("expected number of message / user:" + str(expected_msg) + "\n")
    # file.write("read number of message :" + str(number_msg) + "\n")
    file.write("real number of message / user:" + str(msg_num.value / n) + "\n")

    file.write("Linf error:" + str(error_5) + "\n")
    file.write("50\% error:" + str(error_1) + "\n")
    file.write("90\% error:" + str(error_2) + "\n")
    file.write("95\% error:" + str(error_3) + "\n")
    file.write("99\% error:" + str(error_4) + "\n")
    file.write("average error:" + str(error_6) + "\n")


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    global messages
    global n
    global B
    global b
    global phi
    global eps
    global delta
    global pf
    global phi
    global mu
    global rho
    global data
    global true_frequency
    global levelq
    global bertrand_primes
    global branch
    global error
    global msg_num
    multiprocessing.set_start_method("fork")
    parser = argparse.ArgumentParser(description='optimal small domain range counting for shuffle model')
    parser.add_argument('--dataset', type=str, default='uniform',
                        help='input data set')
    parser.add_argument('--epi', type=float, default=10, help='privacy budget')
    opt = parser.parse_args()
    bertrand_primes = [
        2, 3, 5, 7, 13, 23,
        43, 83, 163, 317, 631, 1259,
        2503, 5003, 9973, 19937, 39869, 79699,
        159389, 318751, 637499, 1274989, 2549951, 5099893,
        10199767, 20399531, 40799041, 81598067, 163196129, 326392249,
        652784471, 1305568919, 2611137817, 5222275627]
    branch = 2
    manager = multiprocessing.Manager()
    messages = manager.dict()
    msg_num = manager.Value(int, 0)
    B = pow(2, 30)
    n = 1e7
    # eps = opt.epi
    eps = 20
    delta = 1 / (n * n)
    s = 0
    t = log2(B)
    c = 2.5
    beta = 0.1
    r = t - s + 1
    b = ceil(n / pow(log2(n), c))
    delta_s = delta / r
    eps_s = eps / r
    mu = 32 * log(2 / delta_s) / (eps_s * eps_s)
    # fixed
    print(mu * b / n, b)
    pre_process()
    # load_data("../uniform.txt")
    process_num = 10
    index = n // 10
    result = []
    global fe
    fe = np.zeros(pow(2, 31))
    # analyzer()
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
        # print(left, right)
        local_data = data[int(left):int(right)]
        result.append(multiprocessing.Process(target=sub_process_randomizer, args=(i, local_data)))
        result[i].start()
    for i in range(process_num):
        result[i].join()

    global total_msg
    total_msg = {}
    for i in range(s, int(t)+1):
        total_msg[i] = []
    for i in range(s, int(t) + 1):
        for j in range(process_num):
            total_msg[i].extend(messages[j][i])
            msg_num.value += len(messages[j][i])

    # for i in range(3):
    #     print(len(messages[i]))
    # result2 = []
    # error = manager.list()
    # for i in range(process_num):
    #     result2.append(multiprocessing.Process(target=sub_process_query, args=(i,)))
    #     result2[i].start()
    # for i in range(process_num):
    #     result2[i].join()
    for i in range(10000):
        l = np.random.randint(0, B)
        h = np.random.randint(0, B)
        while h == l:
            h = np.random.randint(0, B)
        noise_result = range_query(l, h)
        true = true_result(l, h)
        error.append(abs(noise_result - true))
        # fe_counter = collections.Counter(total_msg[i])
        # print(fe_counter)
    global error_1
    global error_2
    global error_3
    global error_4
    global error_5
    global error_6
    # print(np.where(error == max(error)))
    error.sort()
    error_1 = error[int(len(error) * 0.5)]
    error_2 = error[int(len(error) * 0.9)]
    error_3 = error[int(len(error) * 0.95)]
    error_4 = error[int(len(error) * 0.99)]
    error_5 = max(error)
    error_6 = np.average(error)
    out_file = open("../log/Large1D/staL/" + str(opt.rep) + "_eps=" + str(eps) + ".txt", 'w')
    print_info(out_file)
    # print(error_1, error_3)
    print("finish")
    out_file.close()
