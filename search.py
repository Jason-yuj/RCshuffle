import math
import numpy as np


def checker(p, eps, delta, n):
    epow = pow(math.e, eps)
    tp2 = [1.0]
    for i in range(n + 1):
        tp2.append((tp2[i] * (1 - p)))
    pdf = []
    cdf = np.zeros(n)
    C = 1.0
    # calculate pmf
    for i in range(n + 1):
        pdf.append((C * tp2[n - i]))
        C = np.longdouble(C * (n - i) * p / (i + 1))
    # calculate cdf
    cdf[n-1] += pdf[n-1]
    for i in range(n-1, 0, -1):
        cdf[i-1] += (cdf[i] + pdf[i-1])
    # print(cdf)
    prob = 0.0
    # check
    for x_2 in range(n + 1):
        x_1 = math.ceil(epow * x_2 - 1)
        if x_1 < 0:
            x_1 = 0
        if x_1 >= n:
            break
        prob += (np.longdouble(pdf[x_2] * cdf[x_1]))
    return prob <= delta


def search(n, eps, delta):
    le = 0
    # delta = 1 / (n ** 2)
    mu = 32 * math.log(2 / delta) / (eps * eps) / 2
    ri = 1000 / n
    print(32 * math.log(2 / delta) / (eps * eps))
    while le + 1 / n < ri:
        mi = (le + ri) * .5
        print(le, ri)
        if checker(mi, eps, delta, n):
            ri = mi
        else:
            le = mi
    return ri * n


if __name__ == '__main__':
    # result = search(100000000, 0.5)
    # print(result)
    n = int(1e8)
    eps = 0.5
    delta = 1 / (n)
    # p = 5000 / n
    # mu = 32 * math.log(2 / delta) / (eps * eps)
    # print(mu)
    search(n, eps, delta)
    # test = checker(p, eps, delta, n)
    # print(test)
    # t = 0
    # for i in range(10000000):
    #     t += 1
    # print(t)