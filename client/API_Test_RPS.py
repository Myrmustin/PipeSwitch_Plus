
import sys
import time
import struct
import statistics

from task.helper import get_data
from util.util import TcpClient, timestamp
import pickle
import os

def main():
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    model_name_list = model_name.split(';')
    latency_list = []
    
    
    
    for i in range(10):
        print('_____________RUN NUMBER ' + str(i) + ' __________')
        latency = inference(model_name_list,batch_size)
        latency_list.append(latency)
        time.sleep(1)
        
    
    print('Latency for 100 requests : ' + str(latency_list))
    
def inference(model_name_list,batch_size):
    
    model_name = model_name_list[0]

    print('Curent batch of requests : ' + str(model_name_list))

    data_list = []
    for mod in model_name_list:
        data = get_data(mod, batch_size)
        data_list.append(data)
    
    print('List: ' + str(model_name_list))
    
    latency_list = []
    timestamp('client', 'before_request')

    
    # Connect
    client = TcpClient('localhost', 12345)
    timestamp('client', 'after_connect')
    time_1 = time.time()

    # Serialize data
    task_name = model_name + '_inference'
    task_name_b = task_name.encode()
    task_name_length = len(task_name_b)
    task_name_length_b = struct.pack('I', task_name_length)
    
    data_list_b = []
    for db in data_list:
        data_b = db.numpy().tobytes()
        data_list_b.append(data_b)


    with open('/home/ubuntu/Ross/PipeSwitch_Plus/pipeswitch/savedData.p', 'wb') as f:
        pickle.dump(data_list_b, f)

    data_b = data_list_b[0]
    length = len(data_b)
    length_b = struct.pack('I', length)
    timestamp('client', 'after_serialization')

    print("N: " + task_name + ". CurM: " )

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
        #break
    timestamp('client', 'after_reply')
    time_2 = time.time()

    model_name_length = 0
    model_name_length_b = struct.pack('I', model_name_length)
    client.send(model_name_length_b)
    timestamp('client', 'close_training_connection')
    
    timestamp('**********', '**********')
    latency = (time_2 - time_1) * 1000
    latency_list.append(latency)
    print("Inference request on machine X using model " + model_name + " (" + str(batch_size) + " batchsize) completed for: " + str(latency) + "ms. ")
    os.remove("/home/ubuntu/Ross/PipeSwitch_Plus/pipeswitch/savedData.p")

    #time.sleep(2)
    
if __name__ == '__main__':
    main()
