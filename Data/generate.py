import math

import numpy as np


def generate_uniform(n, B):
    file = open("data.txt", 'w')
    # file.write(str(n)+'\n')
    # file.write(str(B)+'\n')
    for _ in range(n):
        i = np.random.randint(0, B)
        file.write(str(i)+'\n')
    print("finish")
    file.close()


def generate_zipf(n, B):
    file = open("test.txt", 'w')
    # file.write(str(n)+'\n')
    # file.write(str(B)+'\n')
    i = 0
    while i <= n:
        data = np.random.zipf(1.3, 1)
        if data < B:
            file.write(str(data[0]) + '\n')
            i += 1
        print(i)
    print("finish")
    file.close()


def dummy():
    file = open("data1.txt", 'w')
    # file.write(str(n)+'\n')
    # file.write(str(B)+'\n')
    for i in range(128):
        # i = np.random.randint(0, B)
        file.write(str(i)+'\n')
    print("finish")
    file.close()


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


if __name__ == '__main__':
    generate_uniform(1000000, 1024)
    # generate_zipf(int(10e5), pow(2, 16))
    #print(bit_count(np.array([[1,2],[3,4]])))
    #3dummy()
