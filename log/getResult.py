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


def get_central(dataset, n, eps, B):
    # 50
    error1_1 = []
    # 90
    error1_2 = []
    # 95
    error1_3 = []
    # 99
    error1_4 = []
    # 100
    error1_5 = []
    # avg
    error1_6 = []
    # 50
    error2_1 = []
    # 90
    error2_2 = []
    # 95
    error2_3 = []
    # 99
    error2_4 = []
    # 100
    error2_5 = []
    # avg
    error2_6 = []
    for i in range(30):
        log_file = "./Small1D/central/" + str(i) + "_" + str(dataset) + "_B=" + str(B) + "_n=" + str(n) + "_eps=" + str(
            eps) + ".txt"
        # --d --q --r --c --o --debug
        file = open(log_file, 'r')
        a = file.readlines()
        error1_5.append(float(a[3].split(":")[1]))
        error1_1.append(float(a[4].split(":")[1]))
        error1_2.append(float(a[5].split(":")[1]))
        error1_3.append(float(a[6].split(":")[1]))
        error1_4.append(float(a[7].split(":")[1]))
        error1_6.append(float(a[8].split(":")[1]))
        error2_5.append(float(a[9].split(":")[1]))
        error2_1.append(float(a[10].split(":")[1]))
        error2_2.append(float(a[11].split(":")[1]))
        error2_3.append(float(a[12].split(":")[1]))
        error2_4.append(float(a[13].split(":")[1]))
        error2_6.append(float(a[14].split(":")[1]))
        file.close()
    error1_1.sort()
    error1_2.sort()
    error1_3.sort()
    error1_4.sort()
    error1_5.sort()
    error1_6.sort()
    error2_1.sort()
    error2_2.sort()
    error2_3.sort()
    error2_4.sort()
    error2_5.sort()
    error2_6.sort()
    error150 = np.average(error1_1[3:27])
    error190 = np.average(error1_2[3:27])
    error195 = np.average(error1_3[3:27])
    error199 = np.average(error1_4[3:27])
    error1100 = np.average(error1_5[3:27])
    erroravg1 = np.average(error1_6[3:27])
    error250 = np.average(error2_1[3:27])
    error290 = np.average(error2_2[3:27])
    error295 = np.average(error2_3[3:27])
    error299 = np.average(error2_4[3:27])
    error2100 = np.average(error2_5[3:27])
    erroravg2 = np.average(error2_6[3:27])
    out_file = open(
        "./Result/" + "central_" + str(dataset) + "_B=" + str(B) + "_n=" + str(n) + "_eps=" + str(
            eps) + ".txt",
        'w')
    out_file.write("pureDP Linf error:" + str(error1100) + "\n")
    out_file.write("pureDP 50\% error:" + str(error150) + "\n")
    out_file.write("pureDP 90\% error:" + str(error190) + "\n")
    out_file.write("pureDP 95\% error:" + str(error195) + "\n")
    out_file.write("pureDP 99\% error:" + str(error199) + "\n")
    out_file.write("pureDP average error:" + str(erroravg1) + "\n")
    out_file.write("approxDP Linf error:" + str(error2100) + "\n")
    out_file.write("approxDP 50\% error:" + str(error250) + "\n")
    out_file.write("approxDP 90\% error:" + str(error290) + "\n")
    out_file.write("approxDP 95\% error:" + str(error295) + "\n")
    out_file.write("approxDP 99\% error:" + str(error299) + "\n")
    out_file.write("approxDP average error:" + str(erroravg2) + "\n")
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
    get_central("gaussian", "1000000", "10.0", "1024")
