
import sys
import time
import struct
import statistics

from task.helper import get_data
from util.util import TcpClient, timestamp


def main():
model_name = sys.argv[1]
inferencePool = sys.argv[2].split(;)   
print(inferencePool[:])
orderedPool = [] 
for i in 3
    most_common = max(inferencePool, key = inferencePool.count)
    for task in inferencePool:
        if task == most_common :
            orderedPool.append(task)
            inferencePool.remove(task)

print(orderedPool)
    

if __name__ == '__main__':
    main()
