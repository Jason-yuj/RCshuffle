# small domain with additive error
import collections
from scipy.stats import binom

import numpy as np
import math


def load_data():
    global data
    file = open('../Data/data.txt', 'r')
    data = []
    a = file.readlines()
    for i in a:
        data.append(int(i))


def pre_process():
    global true_frequency
    true_frequency = np.zeros(B)
    true_counter = collections.Counter(data)
    for i in range(0, B + 1):
        if i in true_counter.keys():
            true_frequency[i] += true_counter[i]


def local_randomizer(x, p):
    global number_msg
    messages.append(x)
    number_msg += 1
    for i in range(0, B):
        if np.random.binomial(1, p):
            messages.append(i)
            number_msg += 1


def analyzer():
    global fe_frequency
    fe_counter = collections.Counter(messages)
    fe_frequency = np.zeros(B)
    for i in range(0, B):
        if i in fe_counter.keys():
            fe_frequency[i] += fe_counter[i]
            # debias-step
            fe_frequency[i] -= mu_1


def true_result(l, h):
    result = 0
    for index in range(min(l, h), max(l, h)):
        result += true_frequency[index]
    return result


def range_query(l, h):
    result = 0
    for index in range(min(l, h), max(l, h)):
        result += fe_frequency[index]
    return result


def print_info(file):
    file.write("epsilon:" + str(eps) + "\n")
    file.write("delta:" + str(delta) + "\n")
    file.write("number of participants:" + str(n) + "\n")
    file.write("domain size:" + str(B) + "\n")
    file.write("mu:" + str(mu_1) + "\n")

    file.write("expected number of message / user:" + str(expected_msg) + "\n")
    # file.write("read number of message :" + str(number_msg) + "\n")
    file.write("real number of message / user:" + str(number_msg / n) + "\n")

    file.write("L1 error:" + str(l1_error) + "\n")
    file.write("L2 error:" + str(l2_error) + "\n")
    file.write("Linf error:" + str(error_5) + "\n")
    file.write("50\% error:" + str(error_1) + "\n")
    file.write("90\% error:" + str(error_2) + "\n")
    file.write("95\% error:" + str(error_3) + "\n")
    file.write("99\% error:" + str(error_4) + "\n")

    # file.write("local randomizer time cost:" + str(eps) + "\n")
    # file.write("analyzer time cost:" + str(eps) + "\n")
    # file.write("single element query time cost:" + str(eps) + "\n")
# def binomcdf():
#     p = 0.5
#     n = 10000
#     x = 0
#     for a in range(10):
#         print(binom.cdf(x, n, p))
#         print(binom.pmf(x, n, p))
#         x += 1000


if __name__ == '__main__':
    global eps
    global delta
    global n
    global B
    global data
    global messages
    global fe_frequency
    global true_frequency
    global number_msg
    global mu_1
    global expected_msg
    B = 1024
    n = 100000
    eps = 5
    delta = 1 / (n * n)
    mu_1 = 32 * math.log(2 / delta) / (eps * eps)
    number_msg = 0
    messages = []
    # print(mu_1 / n)
    # fe_frequencyn= np.zeros(B)
    print("preprocess")
    load_data()
    pre_process()
    # opt_mu = search()
    sample_prob = mu_1 / n
    # print(mu_1, opt_mu)
    expected_msg = 1 + sample_prob * B
    print(sample_prob)
    print("initialize")
    for j in range(n):
        local_randomizer(data[j], sample_prob)
    analyzer()
    error = []
    # all
    global l1_error
    global l2_error
    l2_error = 0.0
    for l in range(B):
        for h in range(l + 1, B):
            # l = np.random.randint(0, B)
            # h = np.random.randint(0, B)
            # while h == l:
            #     h = np.random.randint(0, B)
            noise_result = range_query(l, h)
            true = true_result(l, h)
            error.append(abs(noise_result - true))
            l2_error += pow(abs(noise_result - true), 2)
    error.sort()
    global error_1
    global error_2
    global error_3
    global error_4
    global error_5
    error_1 = error[int(len(error) * 0.5)]
    error_2 = error[int(len(error) * 0.9)]
    error_3 = error[int(len(error) * 0.95)]
    error_4 = error[int(len(error) * 0.99)]
    error_5 = max(error)
    l1_error = sum(error)
    # error_4 = error[int(len(error) * 0.96)]
    # error_5 = error[int(len(error) * 0.97)]
    # error_6 = error[int(len(error) * 0.98)]
    # error_7 = error[int(len(error) * 0.99)]
    # print(number_msg)
    # print(error_1, error_2, error_3, error_4, error_5, error_6, error_7)
    # # print(fe_frequency)
    out_file = open("../log/FE_B=" + str(B) + "_n=" + str(n) + "_eps=" + str(eps) + ".txt", 'w')
    print_info(out_file)
    out_file.close()
    print("finish")
