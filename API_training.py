import sys
import time
import struct
import statistics

from task.helper import get_data
from util.util import TcpClient, timestamp

def send_request(client, task_name, data):
    timestamp('client', 'before_request_%s' % task_name)

    # Serialize data
    task_name_b = task_name.encode()
    task_name_length = len(task_name_b)
    task_name_length_b = struct.pack('I', task_name_length)

    if data is not None:
        data_b = data.numpy().tobytes()
        length = len(data_b)
    else:
        data_b = None
        length = 0
    length_b = struct.pack('I', length)
    timestamp('client', 'after_inference_serialization')

    # Send Data
    client.send(task_name_length_b)
    client.send(task_name_b)
    client.send(length_b)
    if data_b is not None:
        client.send(data_b)
    timestamp('client', 'after_request_%s' % task_name)

def main():
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    print('Start ')
    task_name_inf = '%s_inference' % model_name
    task_name_train = '%s_training' % model_name

    # Load image
    data = get_data(model_name, batch_size)

    # Send training request
    time_1 = time.time()
    client_train = TcpClient('localhost', 12345)
    send_request(client_train, task_name_train, None)
   
    #Recieve responce from training
    recv_response(client_train)
    close_connection(client_train)
    time_2 = time.time()

    latency = (time_2 - time_1) * 1000
    print("Training of " + model_nameon + " on machine X completed for: " + latency + "ms. ")

if __name__ == '__main__':
    main()