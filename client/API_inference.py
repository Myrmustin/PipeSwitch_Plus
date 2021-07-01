import sys
import time
import struct
import statistics

from task.helper import get_data
from util.util import TcpClient, timestamp

def main():
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    model_name_list = model_name.split(';')
    
    
    print('List: ' + str(model_name_list))
    previous_model = ''
    latency_list = []

    for cur_model in model_name_list:
        if(previous_model==''):
            previous_model = cur_model
            latency = regularSend(cur_model, batch_size)
            latency_list.append(latency)
            continue
        

        if(cur_model != previous_model):
            previous_model = cur_model
            latency = regularSend(cur_model, batch_size)
            latency_list.append(latency)
        else:
            latency = requestAwareSend(cur_model,batch_size)
            latency_list.append(latency)

    print("Latency for all requests: " + str(latency_list))

def regularSend(cur_model,batch_size):
    timestamp('client', 'before_request (regularSend)')
    type = 'regularSend'
    type_b= type.encode()
    type_len = len(type_b)
    type_len_b = struct.pack('I', type_len)

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

    print("N: " + task_name)
    print("CurM: " + cur_model)
        

    # Send Data
    client.send(type_len_b)
    client.send(type_b)
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
        #break
    timestamp('client', 'after_reply')
    time_2 = time.time()

    model_name_length = 0
    model_name_length_b = struct.pack('I', model_name_length)
    client.send(model_name_length_b)
    timestamp('client', 'close_training_connection')
        
    timestamp('**********', '**********')
    latency = (time_2 - time_1) * 1000
    print("Inference request on machine X using model " + cur_model + " (" + str(batch_size) + " batchsize) completed for: " + str(latency) + "ms. ")
    #time.sleep(2)
    return latency
def requestAwareSend(cur_model,batch_size):
    timestamp('client', 'before_request (requestAwareSend)')
    type = 'requestAwareSend'
    type_b= type.encode()
    type_len = len(type_b)
    type_len_b = struct.pack('I', type_len)

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

    print("N: " + task_name)
    print("CurM: " + cur_model)
        

    # Send Data
    client.send(type_len_b)
    client.send(type_b)
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
        #break
    timestamp('client', 'after_reply')
    time_2 = time.time()

    model_name_length = 0
    model_name_length_b = struct.pack('I', model_name_length)
    client.send(model_name_length_b)
    timestamp('client', 'close_training_connection')
        
    timestamp('**********', '**********')
    latency = (time_2 - time_1) * 1000
    print("Inference request on machine X using model " + cur_model + " (" + str(batch_size) + " batchsize) completed for: " + str(latency) + "ms. ")
    #time.sleep(2)
    return latency

if __name__ == '__main__':
    main()