import platform
import subprocess
import os
import re
import numpy as np

def get_gpu_usage(ranks):
    exec_nvidia_smi = 'nvidia-smi' if platform.system() == 'Linux' else '\"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe\"'
    pl_output = subprocess.Popen(exec_nvidia_smi, shell=True,
        stdout=subprocess.PIPE, stderr=open(os.devnull, 'w')).stdout.read()

    pattern = re.compile(r'(?P<num>[0-9]{1,5})MiB[\s]+/')
    gpu_mems_usages = []
    for line in pl_output.split('\n'):
        result = pattern.search(line)
        if result:
            gpu_mems_usages.append(int(result.group("num")))
    sorted_gpu_ids = np.argsort(np.array(gpu_mems_usages,dtype= np.float32))
    top = min(ranks, len(gpu_mems_usages))
    return (np.array(range(len(gpu_mems_usages)), dtype= np.int)[sorted_gpu_ids[:top]]).tolist()