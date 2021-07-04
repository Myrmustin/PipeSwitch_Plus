import sys
import time
import struct
import statistics

from task.helper import get_data
from util.util import TcpClient, timestamp


def main():
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2]) 
    latency_list = []
    for i in range(100):
        latency = inference(model_name,batch_size)
        latency_list.append(latency)
        time.sleep(0.5)
    print('Latency for all runs: ' + str(latency_list) )    
def inference(model_name, batch_size):
    model_name_list = model_name.split(';')
    
    
    print('List: ' + str(model_name_list))
    time_o = time.time()
    latency_list = []
    for cur_model in model_name_list:
        timestamp('client', 'before_request')

        data = get_data(cur_model, batch_size)
        
        # Connect
        client = TcpClient('localhost', 12345)
        timestamp('client', 'after_connect')
        time_1 = time.time()

        # Serialize data
        task_name = cur_model + '_inference'
        task_name_b = task_name.encode()
        task_name_length = len(task_name_b)
        task_name_length_b = struct.pack('I', task_name_length)
        data_b = data.numpy().tobytes()
        length = len(data_b)
        length_b = struct.pack('I', length)
        timestamp('client', 'after_serialization')

        print("N: " + task_name + ". CurM: " + cur_model)

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
            break
        timestamp('client', 'after_reply')
        time_2 = time.time()

        model_name_length = 0
        model_name_length_b = struct.pack('I', model_name_length)
        client.send(model_name_length_b)
        timestamp('client', 'close_training_connection')
        
        timestamp('**********', '**********')
        latency = (time_2 - time_1) * 1000
        latency_list.append(latency)
        print("Inference request on machine X using model " + cur_model + " (" + str(batch_size) + " batchsize) completed for: " + str(latency) + "ms. ")
        
        #time.sleep(2)
    time_p = time.time()
    return (time_p - time_o)*1000

if __name__ == '__main__':
    main()