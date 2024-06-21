import collections
import math

import numpy as np
from tqdm import tqdm


def load_data():
    global data
    file = open('../Data/data.txt', 'r')
    data = []
    a = file.readlines()
    for i in a:
        data.append(int(i))


def pre_process():
    global true_frequency
    global true_value
    true_value = []
    for value in data:
        true_value.append(value)
    true_frequency = np.zeros(B)
    true_counter = collections.Counter(true_value)
    # print(counter)
    # true_frequency = np.ones(B)
    for i in range(0, B):
        if i in true_counter.keys():
            true_frequency[i] += true_counter[i]


def local_randomizer(x, p):
    global number_msg
    global messages
    messages.append(B + x - 1)
    # number_msg += 1
    # total_noise = np.random.binomial(2 * B - 1, p)
    # # number_msg += total_noise
    # noise_msg_1 = np.random.randint(0, 2 * B, size=total_noise)
    # noise_msg_2 = noise_msg_1[noise_msg_1 != 0]
    # noise_msg_2 = ((noise_msg_2 + 1) // 2) - 1
    # # number_msg += len(noise_msg_2)
    # # print(noise_msg_1)
    # # print(noise_msg_2)
    # messages += noise_msg_1.tolist()
    # messages += noise_msg_2.tolist()
    noise_msg_1 = np.random.binomial(1, p, size=2 * B - 1)
    noise_msg_1 = np.where(noise_msg_1 == 1)[0]
    noise_msg_2 = noise_msg_1[noise_msg_1 != 0]
    noise_msg_2 = ((noise_msg_2 + 1) // 2) - 1
    messages += noise_msg_1.tolist()
    messages += noise_msg_2.tolist()
    # global test
    # test += noise_msg_1.tolist().count(31)
    # test += noise_msg_2.tolist().count(31)
    # print(test)
    # for i in range(2*B-2, -1, -1):
    #     # np.random.binomial(1, p, size=2*B-1)
    #     if np.random.binomial(1, p):
    #         messages.append(i)
    #         number_msg += 1
    #         if i != 0:
    #             # print(i, i//2)
    #             messages.append((i+1)//2-1)
    #             number_msg += 1
    return


def analyzer():
    global opt_frequency
    global messages
    opt_frequency = np.zeros(2 * B - 1)
    fe_counter = collections.Counter(messages)
    # print(fe_counter)
    for i in range(0, 2 * B - 1):
        if i in fe_counter.keys():
            opt_frequency[i] += fe_counter[i]
            # debias-step
            # rqt_frequency[i] -= mu_1
    # print(opt_frequency)
    # post-processing
    # print(opt_frequency)
    for i in range(B - 1, 0, -1):
        t = pow(-1, math.floor(math.log2(max(1, i))) + (math.log2(B) % 2))
        # print(i, t)
        # print(i-1, 2*i-1, 2*i)
        opt_frequency[i - 1] = t * (opt_frequency[(i - 1)]) + (opt_frequency[2 * i - 1] + opt_frequency[2 * i])
        # print(opt_frequency[(i-1)], (i-1), t)
        # opt_frequency[i-1] -= t * mu_1
    # print(opt_frequency)
    for i in range(1, 2 * B):
        t = pow(-1, math.floor(math.log2(max(1, i))) + (math.log2(B) % 2))
        opt_frequency[i-1] -= t * mu_1
    # print(opt_frequency)
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
    result = 0
    for index in range(min(l, h), max(l, h)):
        result += true_frequency[index]
    return result


def print_info(file):
    file.write("epsilon:" + str(eps) + "\n")
    file.write("delta:" + str(delta) + "\n")
    file.write("number of participants:" + str(n) + "\n")
    file.write("domain size:" + str(B) + "\n")
    file.write("mu:" + str(mu_1) + "\n")

    file.write("expected number of message / user:" + str(expected_msg) + "\n")
    # file.write("read number of message :" + str(number_msg) + "\n")
    file.write("real number of message / user:" + str(len(messages) / n) + "\n")

    file.write("L1 error:" + str(l1_error) + "\n")
    file.write("L2 error:" + str(l2_error) + "\n")
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
    global true_value
    global true_frequency
    global opt_frequency
    global n
    global mu_1
    global test
    test = 0.0
    B = 1024
    n = 1000000
    delta = 1 / (n * n)
    eps = 5
    delta_s = delta / math.log2(B)
    eps_s = eps / math.log2(B)
    mu_1 = 417.968
    number_msg = 0
    messages = []
    print("preprocess")
    load_data()
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
    # true_frequency = np.ones(B)
    # for _ in range(100):
    global l1_error
    global l2_error
    l2_error = 0.0
    for l in tqdm(range(B)):
        for h in range(l + 1, B):
            # l = np.random.randint(0, B)
            # h = np.random.randint(0, B)
            # while h == l:
            #     h = np.random.randint(0, B)
            noise_result = range_query(l, h)
            true = true_result(l, h)
            # print(l,h,noise_result,true)
            # print(l, h, noise_result, true)
            error.append(abs(noise_result - true))
            l2_error += pow(abs(noise_result - true), 2)
    print(opt_frequency)
    # print(error)
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
    l1_error = sum(error)
    error_6 = np.average(error)
    out_file = open("../log/opt_B=" + str(B) + "_n=" + str(n) + "_eps=" + str(eps) + ".txt", 'w')
    print_info(out_file)
    # print(error_1, error_3)
    print("finish")
    out_file.close()
    # print(frequency_1, frequency_2)
