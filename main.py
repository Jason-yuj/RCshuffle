import os
from pathlib import Path
import subprocess
from tqdm import tqdm


def get_project_root() -> Path:
    return Path(__file__).parent

def optS():
    epi_list = ["5", "10", "20"]
    n_list = ["1000000", "10000000", "100000000"]
    b_list = ["512", "1024", "2048"]
    dataset_list = ["uniform", "gaussian", "zipf", "AOL", "netflix"]

    root = get_project_root()
    for n in n_list:
        for b in b_list:
            # generate first
            for dataset in dataset_list:
                if dataset == "AOL" or dataset == "netflix":
                    pass
                else:
                    program_file = str(root) + "/Data/generate.py"
                    cmd = "python " + program_file + " --n " + n + " --b " + b + " --dataset " + dataset
                    subprocess.run(cmd, shell=True)
                for epi in epi_list:
                    for i in range(10):
                        program_file = str(root) + "/OPT/optS.py"
                        cmd = "python " + program_file + " --n " + n + " --b " + b + " --dataset " + dataset + " --epi " + epi + " -- rep " + i
                        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    optS()