import argparse
import collections
from math import log, log2, ceil, floor
import multiprocessing
from bisect import bisect_right, bisect_left

import sys
import numpy as np
from tqdm import tqdm


def load_data(filename):
    global data
    global B
    file = open(filename, 'r')
    data = []
    a = file.readlines()
    # random mapping
    c = np.random.randint(0, B)
    m = np.random.randint(0, B)
    while m % 2 == 0:
        m = np.random.randint(0, B)
    for i in a:
        data.append(int(((int(i) + c) * m) % B))
        # data.append(int(i))


def pre_process():
    global true_frequency
    global data
    global B
    global levelq
    global b
    global bertrand_primes
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


# heavy hitter detection
def randomizer_hhd(x):
    global rho
    global levelq
    global t
    global b
    l = np.random.randint(s, t + 1)
    # l = int(t)
    # small domain
    domain_size = pow(2, l)
    # delta_l = delta / 2
    # mu_l = 32 * math.log(2 / delta_l) / (eps * eps)
    mu_l = mu
    x_p = floor(x / pow(2, t - l))
    # # print(mu_l, mu_l * b / fen)
    # start = x_p*pow(2, t-l)
    # end = x_p*pow(2, t-l) + pow(2, t-l)-1
    # check = start <= x <= end
    # if l == 0:
    #     print(domain_size, x, x_p, check)
    # small domain
    if domain_size < b:
        if np.random.binomial(1, rho):
            messages[l].append(x_p)
        sample_msg = mu_l / fen
        noise_msg_1 = np.random.binomial(1, sample_msg, size=domain_size)
        noise_msg_1 = np.where(noise_msg_1)[0]
        for i in noise_msg_1:
            if np.random.binomial(1, rho):
                messages[l].append(i)
        # fixed_msg = floor(sample_msg)
        # remaining_msg = sample_msg - fixed_msg
        # if np.random.binomial(1, rho):
        #     messages[l].append(x_p)
        # send_msg = fixed_msg + np.random.binomial(1, remaining_msg)
        # for i in range(send_msg):
        #     if np.random.binomial(1, rho):
        #         messages[l].append(np.random.randint(0, domain_size))
    # large domain
    else:
        sample_msg = mu_l * b / fen
        fixed_msg = floor(sample_msg)
        remaining_msg = sample_msg - fixed_msg
        send_msg = fixed_msg + np.random.binomial(1, remaining_msg)
        q = levelq[l]
        if np.random.binomial(1, rho):
            u = np.random.randint(0, q - 1)
            v = np.random.randint(0, q)
            w = ((u * x_p + v) % q) % b
            messages[l].append((u, v, w))
        for i in range(send_msg):
            if np.random.binomial(1, rho):
                u = np.random.randint(0, q - 1)
                v = np.random.randint(0, q)
                w = np.random.randint(0, b)
                messages[l].append((u, v, w))
        # print(sample_msg)
        # print(messages[l])


def quick_power(x, y, mod):
    res = 1
    while y:
        if y & 1:
            res = res * x % mod
        x = x * x % mod
        y >>= 1
    return res


def counter_single(level, id):
    global messages
    global levelq
    global b
    domain_size = pow(2, level)
    frequency = 0
    if domain_size < b:
        i_counter = collections.Counter(messages[level])
        frequency = i_counter[id]
    else:
        q = levelq[level]
        for m in messages[level]:
            u = m[0]
            v = m[1]
            w = m[2]
            if ((u * id + v) % q) % b == w:
                frequency += 1
    return frequency


def counter_all(level):
    global messages
    global levelq
    global b
    domain_size = int(pow(2, level))
    i_frequency = np.zeros(domain_size)
    # test = np.zeros(domain_size)
    # small domain
    if domain_size < b:
        i_counter = collections.Counter(messages[level])
        for i in range(0, domain_size):
            if i in i_counter.keys():
                i_frequency[i] += i_counter[i]
    # large domain
    else:
        q = levelq[level]
        # print(q, b)
        # check = messages[level][0]
        for m in messages[level]:
            u = m[0]
            v = m[1]
            w = m[2]
            invu = quick_power(u, q - 2, q)
            # print(u, v, w)
            # print(invu, u, (invu*u) % q)
            start_id = invu * (w - v + q) % q
            adding = invu * b % q
            # print(start_id, adding, start_id+adding)
            id = start_id

            for j in range(0, ceil((q - 1 - w) / b), 1):
                if 0 <= id < domain_size:
                    i_frequency[id] += 1
                    # if ((u * id + v) % q) % b != w:
                    #     print(id)
                    # print(((u * id + v) % q) % b, w)
                id += adding
                # print(id, "test")
                if id >= q:
                    id -= q
            # print("===============")
            # for i in range(domain_size):
            #     if ((u * i + v) % q) % b == w:
            #         test[i] += 1
            #         # print(i)
    return i_frequency.tolist()


def DomainReduction():
    global rho
    global phi
    global r
    global B
    global s
    global t
    threshold = phi * rho / (2 * r)
    print(threshold)
    # i_frequency = counter(12)
    # print(i_frequency)
    potids = {0: [0]}
    # qryids = {}
    T = []
    cur = 0
    # freq_i = {0: counter_single(0, 0)}
    # next_freq = {}
    full = False
    while cur <= t:
        potids[cur + 1] = []
        # t1 = 2 * len(potids[cur]) * b
        # t2 = pow(2, cur)
        if 2 * len(potids[cur]) * b >= pow(2, cur):
            qryids = counter_all(cur + 1)
            full = True
        # next = []
        for id in potids[cur]:
            if cur == t:
                T.append((cur , id))
                continue
            # freq = freq_i[id]
            # if freq >= threshold:
            if full:
                count_1 = qryids[id * 2]
                count_2 = qryids[id * 2 + 1]
            else:
                count_1 = counter_single(cur + 1, id * 2)
                count_2 = counter_single(cur + 1, id * 2 + 1)
            t_1 = count_1 >= threshold

            t_2 = count_2 >= threshold
            # both child is heavy
            if t_1 and t_2:
                potids[cur + 1].append(id * 2)
                potids[cur + 1].append(id * 2 + 1)
                # next_freq[id * 2] = count_1
                # next_freq[id * 2 + 1] = count_2
            # only child_1 is heavy
            elif t_1:
                potids[cur + 1].append(id * 2)

                T.append((cur + 1, id * 2 + 1))
                # next_freq[id * 2] = count_1
            # only child_2 is heavy
            elif t_2:
                potids[cur + 1].append(id * 2 + 1)
                T.append((cur + 1, id * 2))
                # next_freq[id * 2 + 1] = count_2
            # none of its children is heavy
            else:
                T.append((cur, id))
        # else:
        #     potids[cur].remove(id)
        # freq_i = next_freq.copy()
        # next_freq = {}
        # print(cur, len(potids[cur]))
        # print(len(T))
        full = False
        cur += 1
    # return
    # for (level, pos) in T:
    #     print(level, pos, frequency[level][pos])
    # print(T)
    print(len(T))
    return T


def domain_map_all(T):
    global t
    small_domain = []
    for (level, pos) in T:
        start = pos*pow(2, t-level)
        end = pos*pow(2, t-level) + pow(2, t-level)
        small_domain.append((start, end))
    # sort by its starting point
    small_domain.sort(key=lambda x: x[0])
    return small_domain


def domain_map_single(small_domain_l, small_domain_r, value, flag):
    # flag is for finding left points
    if flag:
        index = bisect_right(small_domain_l, value) - 1
        value = small_domain_l[index]
    else:
        index = bisect_left(small_domain_r, value)
        value = small_domain_l[index]
    return index, value


def sub_process(i, local, p):
    global messages_2
    global small_domain_l
    global small_domain_r
    np.random.seed()
    msg = []
    for d in tqdm(local):
        index_i, _ = domain_map_single(small_domain_l, small_domain_r, d, 1)
        local_msg = randomizer_rc(index_i, p)
        msg.extend(local_msg)
    messages_2[i] = msg


def randomizer_rc(x, p):
    global next
    global messages_2
    global b_1
    global removed_domain
    local_msg = [next + x - 1]
    noise_msg_1 = np.random.binomial(1, p, size=2 * next - 1)
    noise_msg_1[removed_domain] = 0
    noise_msg_1 = np.where(noise_msg_1 == 1)[0]
    noise_msg_2 = noise_msg_1[noise_msg_1 != 0]
    noise_msg_2 = ((noise_msg_2 + 1) // 2) - 1
    local_msg += noise_msg_1.tolist()
    local_msg += noise_msg_2.tolist()
    return local_msg


def analyzer():
    global small_frequency
    global messages_2
    global next
    global total_msg
    total_msg = []
    for i in messages_2.values():
        # print(len(i))
        total_msg.extend(i)
    small_frequency = np.zeros(2 * next - 1)
    fe_counter = collections.Counter(total_msg)
    for i in range(0, 2 * next - 1):
        if i in fe_counter.keys():
            small_frequency[i] += fe_counter[i]
    for i in range(next - 1, 0, -1):
        t = pow(-1, floor(log2(max(1, i))) + (log2(next) % 2))
        small_frequency[i - 1] = t * (small_frequency[(i - 1)]) + (small_frequency[2 * i - 1] + small_frequency[2 * i])
    # debias
    for i in range(1, 2 * next):
        t = pow(-1, floor(log2(max(1, i))) + (log2(next) % 2))
        small_frequency[i - 1] -= t * mu_1
    return


def get_node(B, l, r):
    nodes = []
    start = l
    next = l
    base = int(log2(B)) + 1
    level = int(log2(B)) + 1
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
                level = int(log2(B)) + 1
                start = save_next + 1
                index = start
                next = start
        else:
            segment.append((start, next, level, index))
            level = int(log2(B)) + 1
            start = next + 1
            index = next + 1
            next += 1
    for (_, _, level, index) in segment:
        nodes.append(2 ** (level - 1) + index - 1)
        # print()
    # print(segment)
    return nodes


def range_query(l, h):
    global small_frequency
    nodes = get_node(next, min(l, h), max(l, h))
    # print(nodes)
    result = 0
    for node in nodes:
        result += small_frequency[node]
    return result


def true_result(l, h):
    global data
    index_l = bisect_left(data, min(l, h))
    index_r = bisect_left(data, max(l, h))
    result = index_r - index_l
    return result


def print_info(file):
    file.write("epsilon:" + str(eps) + "\n")
    file.write("epsilon1:" + str(eps_1) + "\n")
    file.write("epsilon2:" + str(eps_2) + "\n")
    file.write("delta:" + str(delta) + "\n")
    file.write("number of participants:" + str(n) + "\n")
    file.write("large domain size:" + str(B) + "\n")
    file.write("reduced small domain:" + str(b_1) + "\n")
    file.write("truncation threshold:" + str(phi) + "\n")
    file.write("expected number of message / user:" + str(expected_msg) + "\n")
    # file.write("reduced domain:" + str(small_domain) + "\n")
    # file.write("mu:" + str(mu_1) + "\n")

    # file.write("expected number of message / user:" + str(expected_msg) + "\n")
    file.write("reduced domain:" + str(small_domain) + "\n")
    file.write("number of message / user for domain reduction:" + str(rd_msg / n) + "\n")
    file.write("number of message / user for estimation:" + str((number_msg - rd_msg) / n) + "\n")

    file.write("Linf error:" + str(total_error_5) + "\n")
    file.write("Linf truncation error:" + str(trunc_error_5) + "\n")
    file.write("Linf second error:" + str(estim_error_5) + "\n")
    file.write("50\% error:" + str(total_error_1) + "\n")
    file.write("50\% truncation error:" + str(trunc_error_1) + "\n")
    file.write("50\% second error:" + str(estim_error_1) + "\n")
    file.write("90\% error:" + str(total_error_2) + "\n")
    file.write("90\% truncation error:" + str(trunc_error_2) + "\n")
    file.write("90\% second error:" + str(estim_error_2) + "\n")
    file.write("95\% error:" + str(total_error_3) + "\n")
    file.write("95\% truncation error:" + str(trunc_error_3) + "\n")
    file.write("95\% second error:" + str(estim_error_3) + "\n")
    file.write("99\% error:" + str(total_error_4) + "\n")
    file.write("99\% truncation error:" + str(trunc_error_4) + "\n")
    file.write("99\% second error:" + str(estim_error_4) + "\n")
    # file.write("average error:" + str(error_6) + "\n")


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    global messages
    global n
    global B
    global b
    global phi
    global eps
    global delta
    global s
    global t
    global pf
    global mu
    global fen
    global rho
    global data
    global true_frequency
    global levelq
    global bertrand_primes
    multiprocessing.set_start_method("fork")
    parser = argparse.ArgumentParser(description='optimal small domain range counting for shuffle model')
    parser.add_argument('--dataset', type=str, default='uniform',
                        help='input data set')
    parser.add_argument('--epi', type=float, default=10, help='privacy budget')
    parser.add_argument('--rep', type=int)
    opt = parser.parse_args()
    # precomputed primes
    bertrand_primes = [
        2, 3, 5, 7, 13, 23,
        43, 83, 163, 317, 631, 1259,
        2503, 5003, 9973, 19937, 39869, 79699,
        159389, 318751, 637499, 1274989, 2549951, 5099893,
        10199767, 20399531, 40799041, 81598067, 163196129, 326392249,
        652784471, 1305568919, 2611137817, 5222275627]
    messages = {}
    data = opt.dataset
    # fixed n and B
    if data == "AOL":
        B = pow(2, 30)
        n = 1e7
    else:
        B = pow(2, 30)
        n = 1e7
    eps = opt.epi
    eps_1 = 6
    eps_2 = eps - eps_1
    delta = 1 / (n * n)
    delta_s = delta / 2
    s = 0
    t = log2(B)
    c = 2
    beta = 0.1
    b = ceil(n / pow(log2(n), c))
    mu = 32 * log(2 / delta_s) / (eps_1 * eps_1)
    print(pow(log(b), 3))
    # fixed
    phi = 8000
    r = t - s + 1
    fen = n / (2 * r)
    print(n, r, fen)
    print(mu * b / fen)
    # exit()
    # rho = (30 * r / phi) * math.log((phi * B / n) + (r * n / (phi * beta)))
    rho = min((30 * r / phi) * log(phi * B / n), (8 * r / phi) * log(r * n / (phi * beta)))
    in_file = opt.dataset
    if in_file == "uniform":
        file_name = "./uniform.txt"
    elif in_file == "AOL":
        file_name = "./AOL_2.txt"
    elif in_file == "zipf":
        file_name = "./zipf.txt"
    elif in_file == "gaussian":
        file_name = "./gaussian.txt"
    else:
        file_name = "./uniform.txt"
    load_data("../uniform.txt")
    pre_process()
    # print( math.log(25000, math.log2(n)))
    # heavy frequency threshold
    print(rho)
    print(phi)
    print(phi * rho / (2 * r))
    print(b)
    # exit()
    for i in range(s, int(t) + 1):
        messages[i] = []

    # first round for domain reduction
    for i in tqdm(data):
        true_frequency[i] += 1
        randomizer_hhd(i)
    num = 0
    for i in range(s, int(t) + 1):
        num += len(messages[i])
    global rd_msg
    rd_msg = num
    print(num)
    T = DomainReduction()
    # second round for range counting
    global number_msg
    global messages_2
    global small_frequency
    global total_error
    global trunc_error
    global estim_error
    small_domain = domain_map_all(T)
    total_error = []
    trunc_error = []
    estim_error = []
    b_1 = len(small_domain)
    # parameter for range counting
    # round to the power of 2
    global next
    global total_msg
    next = pow(2, ceil(log(b_1) / log(2)))
    delta_s = delta / (log2(next) + 1)
    eps_s = eps_2 / (log2(next) + 1)
    mu_1 = 32 * log(2 / delta_s) / (eps_s * eps_s)
    # mu_1 = 23.0713
    sample_prob = mu_1 / n
    # messages_2 = []
    global small_domain_l
    global small_domain_r
    global expected_msg
    global removed_domain
    removed_domain = []
    extra_domain = np.arange(next + b_1, 2 * next)
    removed_domain.extend(extra_domain - 1)
    while len(extra_domain) != 1:
        if extra_domain[0] % 2 == 1:
            extra_domain = extra_domain[1:]
        extra_domain = np.unique(extra_domain // 2)
        removed_domain.extend((extra_domain - 1).tolist())
    small_domain_l = [d[0] for d in small_domain]
    small_domain_r = [d[1] for d in small_domain]
    process_num = 5
    index = n // process_num
    result = []
    manager = multiprocessing.Manager()
    messages_2 = manager.dict()
    for i in range(process_num):
        # Try to make  parameters locally
        if i < process_num - 1:
            left = index * i
            right = index * (i + 1)
        else:
            left = index * i
            right = n
        # print(i, left, right)
        messages_2[i] = []
        local_data = data[int(left):int(right)]
        result.append(multiprocessing.Process(target=sub_process, args=(i, local_data, sample_prob)))
        result[i].start()
    for i in range(process_num):
        result[i].join()
    # for i in tqdm(data):
    #     index_i, _ = domain_map_single(small_domain_l, small_domain_r, i, 1)
    #     randomizer_rc(index_i, sample_prob)
    analyzer()
    expected_msg = 1 + (4 * next - 3) * sample_prob
    print(len(total_msg))
    number_msg = num + len(total_msg)
    print("finish")
    data.sort()
    # range count
    for i in range(100000):
        l = np.random.randint(0, B)
        h = np.random.randint(0, B)
        while h == l:
            h = np.random.randint(0, B)
        # to small domain
        small_l, truncated_l = domain_map_single(small_domain_l, small_domain_r, min(l, h), 1)
        small_h, truncated_r = domain_map_single(small_domain_l, small_domain_r, max(l, h), 0)
        # print(l, h, truncated_r, truncated_r)
        noise_result = range_query(small_l, small_h)
        true = true_result(l, h)
        truncated_true = true_result(truncated_l, truncated_r)
        # if i <= 10:
        #     print(l, h, small_l, small_h)
        #     print(noise_result, true, abs(noise_result - true))
        total_error.append(abs(noise_result - true))
        trunc_error.append(abs(truncated_true-true))
        estim_error.append(abs(noise_result - truncated_true))
    global error_1
    global error_2
    global error_3
    global error_4
    global error_5
    global error_6
    # print(np.where(error == max(error)))
    total_error.sort()
    total_error_1 = total_error[int(len(total_error) * 0.5)]
    total_error_2 = total_error[int(len(total_error) * 0.9)]
    total_error_3 = total_error[int(len(total_error) * 0.95)]
    total_error_4 = total_error[int(len(total_error) * 0.99)]
    total_error_5 = total_error[-1]
    trunc_error.sort()
    trunc_error_1 = trunc_error[int(len(trunc_error) * 0.5)]
    trunc_error_2 = trunc_error[int(len(trunc_error) * 0.9)]
    trunc_error_3 = trunc_error[int(len(trunc_error) * 0.95)]
    trunc_error_4 = trunc_error[int(len(trunc_error) * 0.99)]
    trunc_error_5 = trunc_error[-1]
    estim_error.sort()
    estim_error_1 = estim_error[int(len(estim_error) * 0.5)]
    estim_error_2 = estim_error[int(len(estim_error) * 0.9)]
    estim_error_3 = estim_error[int(len(estim_error) * 0.95)]
    estim_error_4 = estim_error[int(len(estim_error) * 0.99)]
    estim_error_5 = estim_error[-1]
    # error_6 = np.average(error)
    out_file = open("../log/Large1D/optL/" + str("zipf0.5") + "_eps=" + str(eps) + "_shuffle1.5.txt", 'w')
    print_info(out_file)
    # print(error_1, error_3)
    print("finish")
    out_file.close()
