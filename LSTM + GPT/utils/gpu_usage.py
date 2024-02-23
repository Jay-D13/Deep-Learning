import subprocess
import re


def get_gpu_memory_usage():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
        memory = int(re.search(r'\d+', result.decode('utf-8')).group())
        return memory
    except:
        print("Error while retrieving GPU memory.")
        return 0
