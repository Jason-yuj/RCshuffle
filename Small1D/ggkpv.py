import time
import numpy as np
import math
import collections

from tqdm import tqdm


def load_data():
    global data
    file = open('../Data/data.txt', 'r')
    data = []
    a = file.readlines()
    for i in a:
        data.append(int(i))


# generate hadamard matrix
def matrix(n):
    if n == 1:
        m = [[1, 1], [1, 0]]
        return np.array(m)
    else:
        m = matrix(n - 1)
        return np.vstack((np.hstack((m, m)), np.hstack((m, 1 - m))))


# check whether in hadamard matrix
def in_hadamard(x, y):
    p = x & y
    c = p.bit_count()
    a = pow(-1, c)
    return a


def hadamard_response(x, B):
    r = np.random.randint(0, 2 * B)
    while in_hadamard(x, r) != 1:
        r = np.random.randint(0, 2 * B)
    return r


def pre_process():
    global M
    global M_inverse
    global true_frequency
    M = np.zeros((B, B))
    row = 0
    for i in range(int(math.log2(B) + 1)):
        for j in range(int(B / (2 ** (i + 1)))):
            M[row][(2 ** i) * (2 * j):(2 ** i) * (2 * j) + 2 ** i] += 1
            # print(row,(2**i)*(2*j),2**i)
            row += 1
    M[row] += 1
    # print(M)
    M_inverse = np.linalg.inv(M)
    # return M, M^-1
    for value in data:
        true_value.append(value)
    true_frequency = np.zeros(B)
    true_counter = collections.Counter(true_value)
    # print(counter)
    for i in range(0, B + 1):
        if i in true_counter.keys():
            true_frequency[i] += true_counter[i]
    return


def local_randomizer(x, B, delta, epsilon):
    '''
    :param x: elements
    :param epsilon:
    :param delta:
    :param n: total user
    :param B: domain size
    :param tau: number of message per element
    :return:
    '''
    global messages
    global tau
    global rho
    global k
    global hadamard_m

    aug_k = k - len(x)
    elements = x

    # augment messages
    for i in range(aug_k):
        elements.append(B + i + 1)
    # generate hadamard response
    for e in elements:
        candidate = np.where(hadamard_m[e + 1] == 1)[0]
        message = np.random.choice(candidate, size=tau)
        # message = []
        # for _ in range(tau):
        #     a_i = hadamard_response(e + 1, B)
        #     message.append(a_i)
        messages.append(message)
    # generate noise message
    for _ in range(rho):
        message = np.random.randint(0, 2 * B, size=tau)
        messages.append(message)
    # return all the messages
    # return messages


def bit_count(arr):
    # Make the values type-agnostic (as long as it's integers)
    t = arr.dtype.type
    mask = t(-1)
    s55 = t(0x5555555555555555 & mask)  # Add more digits for 128bit support
    s33 = t(0x3333333333333333 & mask)
    s0F = t(0x0F0F0F0F0F0F0F0F & mask)
    s01 = t(0x0101010101010101 & mask)

    arr = arr - ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)
    arr = (arr + (arr >> 4)) & s0F
    return (arr * s01) >> (8 * (arr.itemsize - 1))


def analyzer(B):
    global frequency
    index = []
    # print(len(messages))
    total = len(messages)
    # c = 0
    for t in tqdm(range(len(messages))):
        a = np.tile(np.arange(1, B+1), (tau, 1)).T
        test = messages[t] & a
        bit = bit_count(test) % 2
        valid = np.sum(bit, axis=1)
        # print((valid))
        #print(len(np.where([valid == 0])))
        index_t = np.where(valid == 0)[0].tolist()
        # print(np.where([valid == 0]))
        index += index_t
        # if t == 0:
        #     for j in range(1, B+1):
        #         print(messages[t] & j)
        #     print(messages[t], a, test, bit, valid, index)
        # for j in range(1, B + 1):
        #     test = messages[t] & j
        #     bit = bit_count(test)
        #
        #     if sum(bit % 2) == 0:
        #     # for i in messages[t]:
        #     #     flag = True
        #     #     if (i & j).bit_count() % 2:
        #     #         flag = False
        #     #         break
        #     #     if flag:
        #             index.append(j - 1)
        # if c % (total / 10) == 0:
        #     print(c / total)
        # c += 1
    # print(index, len(index))
    frequency = np.zeros(B)
    counter = collections.Counter(index)
    # print(counter)
    for i in range(0, B + 1):
        if i in counter.keys():
            frequency[i] += counter[i]
    # print(frequency)
    # de-biasing step
    for i in range(len(frequency)):
        frequency[i] = (1 / (1 - pow(2, -tau))) * (frequency[i] - (rho + k) * n * pow(2, -tau))
    # return B array for all left nodes in range tree


# random range query
def range_query(l, h):
    global M_inverse
    global frequency

    q = np.zeros(B)
    for index in range(min(l, h), max(l, h)):
        q[index] += 1
    tree_q = np.dot(q, M_inverse)
    result = np.dot(tree_q, frequency)
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
    # file.write("mu:" + str(mu_1) + "\n")

    # file.write("read number of message :" + str(number_msg) + "\n")
    file.write("real number of message / user:" + str(msg) + "\n")

    file.write("L1 error:" + str(l1_error) + "\n")
    file.write("L2 error:" + str(l2_error) + "\n")
    file.write("Linf error:" + str(error_5) + "\n")
    file.write("50\% error:" + str(error_1) + "\n")
    file.write("90\% error:" + str(error_2) + "\n")
    file.write("95\% error:" + str(error_3) + "\n")
    file.write("99\% error:" + str(error_4) + "\n")


if __name__ == '__main__':
    global messages
    global tau
    global rho
    global k
    global frequency
    global M
    global M_inverse
    global data
    global true_value
    global true_frequency
    global B
    global hadamard_m
    # print(matrix(4))

    # for i in range(4):
    #     for j in range(4):
    #         print(in_hadamard(i,j), m[i][j])
    # local_randomizer(x, n, B,  delta, epsilon):
    # m = (local_randomizer([0,1,2], 10, 4, 1e-10, 0.5))
    # print(m)
    # print(len(m))
    # time = 0
    # start = time.time()
    # local_randomizer([0,1,1,1], 1024, 1e-10, 4)
    # analyzer(1024)
    # end = time.time()
    # print(end - start)
    Bs = [512, 1024, 2048]
    ns = [1000000, 10000000, 100000000]

    eps = [5,10, 20]
    # messages = []
    # true_value = []
    for B in Bs:
        for n in ns:
            for e in eps:
                delta = 1 / (n * n)
                tau = math.ceil(math.log2(n))
                # max elements per user
                k = math.ceil(math.log2(2 * B))
                # hadamard_m = matrix(k)
                # k = 1
                # number of noise messages
                rho = math.ceil((36.0 * k * k * math.log(math.e * k / (delta * e))) / (e * e))
                msg = rho + k
                print(B, n, e, msg)
    # load_data()
    # pre_process()
    # print("initializing...")
    # # print(M_inverse)
    # start = time.time()
    # c = 0
    # for i in tqdm(range(n)):
    #     # c += 1
    #     # if c % (len(data) / 10) == 0:
    #     #     print(c / len(data))
    #     #     test = time.time()
    #     #     print(test - start)
    #     # x = [i]
    #     vec_i = np.zeros(B)
    #     vec_i[data[i]] += 1
    #     range_q = np.dot(M, vec_i)
    #     x = np.where(range_q == 1)[0].tolist()
    #     # print(x)
    #     local_randomizer(x, B, delta, eps)
    #
    # analyzer(B)
    # # print(true_frequency)
    # print("finish!")
    # # print(frequency)
    # end = time.time()
    # print(end - start)
    # error = []
    # # for _ in range(100):
    # global l1_error
    # global l2_error
    # l2_error = 0.0
    # for l in tqdm(range(B)):
    #     for h in range(l + 1, B):
    #         # l = np.random.randint(0, B)
    #         # h = np.random.randint(0, B)
    #         # while h == l:
    #         #     h = np.random.randint(0, B)
    #         noise_result = range_query(l, h)
    #         true = true_result(l, h)
    #         error.append(abs(noise_result - true))
    #         l2_error += pow(noise_result - true, 2)
    #     # print(noise_result, true)
    # # print(error)
    # global error_1
    # global error_2
    # global error_3
    # global error_4
    # global error_5
    # error.sort()
    # error_1 = error[int(len(error) * 0.5)]
    # error_2 = error[int(len(error) * 0.9)]
    # error_3 = error[int(len(error) * 0.95)]
    # error_4 = error[int(len(error) * 0.99)]
    # error_5 = max(error)
    # error_6 = np.average(error)
    # l1_error = sum(error)
    # out_file = open("../log/ggkpv_B=" + str(B) + "_n=" + str(n) + "_eps=" + str(eps) + ".txt", 'w')
    # # file.write(str(n)+'\n')
    # # file.write(str(B)+'\n')
    # print_info(out_file)
    # print("finish")
    # out_file.close()
    # print(error_1, error_2, error_3, error_4)
    # print(np.max(np.array(error)))
