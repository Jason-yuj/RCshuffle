import time
import numpy as np
import math
import collections
from math import log, log2, ceil, floor, sqrt
from bisect import bisect_right, bisect_left

from tqdm import tqdm


def load_data():
    global data
    file = open('./Data/data.txt', 'r')
    data = []
    a = file.readlines()
    for i in a:
        data.append(int(i))


def pre_process():
    global M
    global M_inverse
    global true_frequency
    M = np.zeros((2*B-1, B))
    row = 0
    for i in range(int(math.log2(B) + 1)):
        for j in range(int(B / (2 ** i))):
            M[row][(2 ** i) * j:(2 ** i) * j + 2 ** i] += 1
            # print(row,(2**i)*(2*j),2**i)
            row += 1
    # M[row] += 1
    # print(M)
    M_inverse = np.linalg.pinv(M)
    # print(M_inverse.shape)
    # return M, M^-1
    true_frequency = np.zeros(B)
    true_counter = collections.Counter(data)
    # print(counter)
    for i in range(0, B + 1):
        if i in true_counter.keys():
            true_frequency[i] += true_counter[i]
    return


def range_query_pure(W, z):
    global M_inverse
    global true_frequency

    q = np.zeros(B)
    for index in range(min(l, h), max(l, h)):
        q[index] += 1
    # tree_q = np.dot(q, M_inverse)
    # # result = np.dot(tree_q, true_frequency)
    sensitivity = max(np.sum(M, axis=0))
    # x_p = np.dot(M, true_frequency) + sensitivity * z
    # result = np.dot(tree_q, x_p)

    t = np.dot(W, true_frequency)
    temp = np.dot(W, M_inverse)
    noise = np.dot(temp, sensitivity * z)
    return t + noise


def range_query_approx(W):
    global M_inverse
    global true_frequency

    q = np.zeros(B)
    for index in range(min(l, h), max(l, h)):
        q[index] += 1
    tree_q = np.dot(q, M_inverse)
    # result = np.dot(tree_q, true_frequency)
    sensitivity = math.sqrt(max(np.sum(M, axis=0)))
    sigma = sqrt(2*log(2/delta))/eps
    z = np.random.normal(0, sigma, size=B)
    x_p = np.dot(M, true_frequency) + sensitivity * z
    result = np.dot(tree_q, x_p)
    return result

def true_result(W):
    # global data
    # index_l = bisect_left(data, min(l, h))
    # index_r = bisect_left(data, max(l, h))
    # result = index_r - index_l
    # return result
    global true_frequency
    return np.dot(W, true_frequency)


def print_info(file):
    file.write("epsilon:" + str(eps) + "\n")
    file.write("delta:" + str(delta) + "\n")
    file.write("number of participants:" + str(n) + "\n")
    file.write("domain size:" + str(B) + "\n")
    # file.write("mu:" + str(mu_1) + "\n")

    # file.write("expected number of message / user:" + str(expected_msg) + "\n")
    # file.write("read number of message :" + str(number_msg) + "\n")
    # file.write("real number of message / user:" + str(number_msg / n) + "\n")

    file.write("L1 error:" + str(l1_error) + "\n")
    file.write("L2 error:" + str(l2_error) + "\n")
    file.write("Linf error:" + str(error_5) + "\n")
    file.write("50\% error:" + str(error_1) + "\n")
    file.write("90\% error:" + str(error_2) + "\n")
    file.write("95\% error:" + str(error_3) + "\n")
    file.write("99\% error:" + str(error_4) + "\n")
    file.write("average error:" + str(error_6) + "\n")


if __name__ == '__main__':
    global B
    global n
    global eps
    global delta
    global error
    global M
    global M_inverse
    global true_frequency
    B = 1024
    n = 1e6
    eps = 5
    delta = 1 / (n ** 2)
    eps /= log2(B)
    delta /= log2(B)
    error = []
    l2_error = 0
    load_data()
    pre_process()
    data.sort()
    z = np.random.laplace(0, 1 / eps, size=2*B-1)
    # range_query_pure(1, 3)
    # W = np.array([])
    for l in tqdm(range(B)):
        W = np.array([])
        for h in range(l + 1, B):
            q = np.zeros(B)
            for index in range(min(l, h), max(l, h)):
                q[index] += 1
            if h == l + 1:
                W = q.copy()
            else:
                # print(W.shape)
                # print(q.shape)
                np.vstack((W,q))

        result = range_query_pure(W, z)
        true = true_result(W)
        # print(result, true, (result - true))
        error += np.abs(result - true)
        # l2_error += pow(result - true, 2)
    error.sort()
    global error_1
    global error_2
    global error_3
    global error_4
    global error_5
    global error_6
    error_1 = error[int(len(error) * 0.5)]
    error_2 = error[int(len(error) * 0.9)]
    error_3 = error[int(len(error) * 0.95)]
    error_4 = error[int(len(error) * 0.99)]
    error_5 = max(error)
    l1_error = sum(error)
    error_6 = np.average(error)
    print(error_1, error_2, error_3, error_4, error_5, error_6)
