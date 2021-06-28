import sys
import time
import struct
import statistics

from task.helper import get_data
from util.util import TcpClient, timestamp

def main():
    model_name = sys.argv[1]
    model_name_list = model_name.split(';')
    batch_size = int(sys.argv[2])

    

    for model_name_o in model_name_list:
        # Load image
        data = get_data(model_name_o, batch_size)
        timestamp('client', 'before_request')

        # Connect
        client = TcpClient('localhost', 12345)
        timestamp('client', 'after_connect for model: ' + model_name_o)
        time_1 = time.time()

        # Serialize data
        task_name = model_name_o + '_inference'
        task_name_b = task_name.encode()
        task_name_length = len(task_name_b)
        task_name_length_b = struct.pack('I', task_name_length)
        data_b = data.numpy().tobytes()
        length = len(data_b)
        length_b = struct.pack('I', length)
        timestamp('client', 'after_serialization')

        # Send Data
        client.send(task_name_length_b)
        client.send(task_name_b)
        client.send(length_b)
        client.send(data_b)
        timestamp('client', 'after_send')

        # Get reply
        reply_b = client.recv(4)
        reply = reply_b.decode()
        if reply == 'FAIL':
            timestamp('client', 'FAIL')
        timestamp('client', 'after_reply')
        time_2 = time.time()

        model_name_length = 0
        model_name_length_b = struct.pack('I', model_name_length)
        client.send(model_name_length_b)
        timestamp('client', 'close_training_connection')

        timestamp('**********', '**********')
        latency = (time_2 - time_1) * 1000
        print("Inference request on machine X using model " + model_name_o + " (" + str(batch_size) + " batchsize) completed for: " + str(latency) + "ms. ")
        

if __name__ == '__main__':
    main()