import os
from pathlib import Path
import subprocess
from tqdm import tqdm


def get_project_root() -> Path:
    return Path(__file__).parent


def optS_n():
    epi = "10"
    n_list = ["1000000", "10000000", "100000000"]
    # n_list = ["1000000"]
    b = "1024"
    # b_list = ["512"]
    dataset_list = ["uniform", "gaussian", "zipf", "AOL", "netflix"]
    # dataset_list = ["AOL", "netflix"]

    root = get_project_root()
    for dataset in dataset_list:
        # generate first
        # if dataset == "AOL" or dataset == "netflix":
        #     for n in n_list:
        #         for i in range(30):
        #             program_file_1 = str(root) + "/Small1D/optS.py"
        #             cmd = "python " + program_file_1 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
        #                 i)
        #             subprocess.run(cmd, shell=True)
        #             program_file_2 = str(root) + "/Small1D/RQT.py"
        #             cmd = "python " + program_file_2 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
        #                 i)
        #             subprocess.run(cmd, shell=True)
        #             program_file_3 = str(root) + "/Small1D/central.py"
        #             cmd = "python " + program_file_3 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
        #                 i)
        #             subprocess.run(cmd, shell=True)
        # else:
        for n in n_list:
            program_file = str(root) + "/Data/generate.py"
            cmd = "python " + program_file + " --n " + n + " --b " + b + " --dataset " + dataset
            subprocess.call(cmd, shell=True)
            for i in range(30):
                program_file_1 = str(root) + "/Small1D/newOPTs.py"
                cmd = "python " + program_file_1 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                                i)
                subprocess.run(cmd, shell=True)
                program_file_2 = str(root) + "/Small1D/newRQT.py"
                cmd = "python " + program_file_2 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                                i)
                subprocess.run(cmd, shell=True)
                program_file_3 = str(root) + "/Small1D/central.py"
                cmd = "python " + program_file_3 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                                i)
                subprocess.run(cmd, shell=True)


def optS_epi():
    epi_list = ["5", "10", "20"]
    # n_list = ["1000000", "10000000", "100000000"]
    n = "10000000"
    b = "1024"
    # b_list = ["512"]
    dataset_list = ["uniform", "gaussian", "zipf", "AOL", "netflix"]
    # dataset_list = ["AOL", "netflix"]

    root = get_project_root()
    for dataset in dataset_list:
        # generate first
        if dataset == "AOL" or dataset == "netflix":
            for epi in epi_list:
                for i in range(30):
                    program_file_1 = str(root) + "/Small1D/optS.py"
                    cmd = "python " + program_file_1 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                            i)
                    subprocess.run(cmd, shell=True)
                    program_file_2 = str(root) + "/Small1D/RQT.py"
                    cmd = "python " + program_file_2 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                            i)
                    subprocess.run(cmd, shell=True)
                    program_file_3 = str(root) + "/Small1D/central.py"
                    cmd = "python " + program_file_3 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                            i)
                    subprocess.run(cmd, shell=True)
        else:
            program_file = str(root) + "/Data/generate.py"
            cmd = "python " + program_file + " --n " + n + " --b " + b + " --dataset " + dataset
            subprocess.call(cmd, shell=True)
            for epi in epi_list:
                for i in range(30):
                    program_file_1 = str(root) + "/Small1D/newOPTs.py"
                    cmd = "python " + program_file_1 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                                i)
                    subprocess.run(cmd, shell=True)
                    program_file_2 = str(root) + "/Small1D/newRQT.py"
                    cmd = "python " + program_file_2 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                                i)
                    subprocess.run(cmd, shell=True)
                    program_file_3 = str(root) + "/Small1D/central.py"
                    cmd = "python " + program_file_3 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                                i)
                    subprocess.run(cmd, shell=True)


def optS_b():
    epi = "10"
    # n_list = ["1000000", "10000000", "100000000"]
    n = "10000000"
    # b = "1024"
    b_list = ["512", "1024", "2048"]
    dataset_list = ["uniform", "gaussian", "zipf", "AOL", "netflix"]
    # dataset_list = ["AOL", "netflix"]

    root = get_project_root()
    for dataset in dataset_list:
        # generate first
        if dataset == "AOL" or dataset == "netflix":
            for b in b_list:
                for i in range(30):
                    program_file_1 = str(root) + "/Small1D/optS.py"
                    cmd = "python " + program_file_1 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                            i)
                    subprocess.run(cmd, shell=True)
                    program_file_2 = str(root) + "/Small1D/RQT.py"
                    cmd = "python " + program_file_2 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                            i)
                    subprocess.run(cmd, shell=True)
                    program_file_3 = str(root) + "/Small1D/central.py"
                    cmd = "python " + program_file_3 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                            i)
                    subprocess.run(cmd, shell=True)
        else:
            for b in b_list:
                program_file = str(root) + "/Data/generate.py"
                cmd = "python " + program_file + " --n " + n + " --b " + b + " --dataset " + dataset
                subprocess.call(cmd, shell=True)
                for i in range(30):
                    program_file_1 = str(root) + "/Small1D/newOPTs.py"
                    cmd = "python " + program_file_1 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                                i)
                    subprocess.run(cmd, shell=True)
                    program_file_2 = str(root) + "/Small1D/newRQT.py"
                    cmd = "python " + program_file_2 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                                i)
                    subprocess.run(cmd, shell=True)
                    program_file_3 = str(root) + "/Small1D/central.py"
                    cmd = "python " + program_file_3 + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " --rep " + str(
                                i)
                    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    optS_b()
