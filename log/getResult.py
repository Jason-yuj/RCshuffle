import os
from pathlib import Path
import subprocess
from tqdm import tqdm
import numpy as np


def get_result(dataset, n, eps, B, type):
    # 50
    error_1 = []
    # 90
    error_2 = []
    # 95
    error_3 = []
    # 99
    error_4 = []
    # 100
    error_5 = []
    # avg
    error_6 = []
    msg = []
    for i in range(30):
        log_file = "./Small1D/" + type + "/" + str(i) + str(dataset) + "_B=" + str(B) + "_n=" + str(n) + "_eps=" + str(
            eps) + "_2.txt"
        # --d --q --r --c --o --debug
        file = open(log_file, 'r')

        a = file.readlines()
        msg.append(float(a[7].split(":")[1]))
        error_5.append(float(a[8].split(":")[1]))
        error_1.append(float(a[9].split(":")[1]))
        error_2.append(float(a[10].split(":")[1]))
        error_3.append(float(a[11].split(":")[1]))
        error_4.append(float(a[12].split(":")[1]))
        error_6.append(float(a[13].split(":")[1]))
        file.close()
    error_1.sort()
    error_2.sort()
    error_3.sort()
    error_4.sort()
    error_5.sort()
    error_6.sort()
    msg.sort()
    error50 = np.arange(error_1[3:27])
    error90 = np.arange(error_2[3:27])
    error95 = np.arange(error_3[3:27])
    error99 = np.arange(error_4[3:27])
    error100 = np.arange(error_5[3:27])
    erroravg = np.arange(error_6[3:27])
    msg_num = np.arange(msg[3:27])
    out_file = open(
        "./Result/" + "newOPTs_" + str(dataset) + "_B=" + str(B) + "_n=" + str(n) + "_eps=" + str(
            eps) + "_2.txt",
        'w')
    out_file.write("real number of message / user:" + str(msg_num) + "\n")
    out_file.write("Linf error:" + str(error100) + "\n")
    out_file.write("50\% error:" + str(error50) + "\n")
    out_file.write("90\% error:" + str(error90) + "\n")
    out_file.write("95\% error:" + str(error95) + "\n")
    out_file.write("99\% error:" + str(error99) + "\n")
    out_file.write("average error:" + str(erroravg) + "\n")
    out_file.close()


def get_n():
    n_list = ["1000000", "10000000", "100000000"]
    B = "1024"
    eps = "10.0"
    dataset_list = ["uniform", "gaussian", "zipf"]
    for dataset in dataset_list:
        for n in n_list:
            get_result(dataset, n, eps, B)


def get_B():
    b_list = ["512", "1024", "2048"]
    eps = "10.0"
    n = "100000000"
    dataset_list = ["uniform", "gaussian", "zipf"]
    for dataset in dataset_list:
        for B in b_list:
            get_result(dataset, n, eps, B, "newOPTs")
            get_result(dataset, n, eps, B, "newRQT")


def get_epi():
    epi_list = ["5.0", "10.0", "20.0"]
    B = "1024"
    n = "100000000"
    dataset_list = ["uniform", "gaussian", "zipf"]
    for dataset in dataset_list:
        for epi in epi_list:
            get_result(dataset, n, epi, B, "newOPTs")
            get_result(dataset, n, epi, B, "newRQT")


def get_real():
    dataset_list = ["AOL", "netflix"]
    epi_list = ["5.0", "10.0", "20.0"]
    for dataset in dataset_list:
        if dataset == "AOL":
            B = "1024"
            n = "1230096"
        else:
            B = "2048"
            n = "99399848"
        for epi in epi_list:
            get_result(dataset, n, epi, B, "newOPTs")
            get_result(dataset, n, epi, B, "newRQT")


if __name__ == "__main__":
    get_B()
