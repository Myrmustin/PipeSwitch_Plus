
import sys
import time
import struct
import statistics

from task.helper import get_data
from util.util import TcpClient, timestamp
import pickle
import os
import numpy as np

def main():
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    model_name_list = model_name.split(';')
    
    #those are hardcoded examples 
    BBcount = [obj for obj in model_name_list if obj == "bert_base"]
    RNcount = [obj for obj in model_name_list if obj == "resnet152"]
    I3count = [obj for obj in model_name_list if obj == "inception_v3"]
    RN101Mcount = [obj for obj in model_name_list if obj == "resnext101_32x8d"]
    RN34count = [obj for obj in model_name_list if obj == "resnet34"]
    RN50Mcount = [obj for obj in model_name_list if obj == "resnext50_32x4d"]
    
    latency_list = []
    
    """for i in range(2):
        print('_____________DRY RUN__________')
        latency1 = inference(BBcount,batch_size)
        latency2 = inference(I3count,batch_size)
        latency3 = inference(RNcount,batch_size)
        latency4 = inference(RN101count,batch_size)
        time.sleep(0.5)"""
    for i in range(50):
        print('_____________RUN NUMBER ' + str(i) + ' __________')
        

        # complex test case 1
        """latency1 = inference(BBcount,batch_size)
        latency2 = inference(I3count,batch_size)
        latency3 = inference(RNcount,batch_size)
        latency4 = inference(RN101Mcount,batch_size)
        total_latency = latency1 + latency2 + latency3 + latency4
        latency_list.append(total_latency)"""
        #complex test case 2
        # 3x Resnet24 | 2x Resnext 50 | 3x Resnet34 | 2x Resnext50 | 3x Resnet34 
        """tmp = np.array_split(RN34count,3)
        tmp2 = np.array_split(RN50Mcount,2)
        latency1 = inference(tmp[0],batch_size)
        latency2 = inference(tmp2[0],batch_size)
        latency3 = inference(tmp[1],batch_size)
        latency4 = inference(tmp2[1],batch_size)
        latency5 = inference(tmp[2],batch_size)
        total_latency = latency1 + latency2 + latency3 + latency4 + latency5
        latency_list.append(total_latency)
        time.sleep(0.5)"""
        
        #complex 3
        #  4x inception 1x Rnext101 4x inception 1x bert_base 4x inception 1x Rnext50 4x inception
        tmp = np.array_split(I3count,4)
        latency1 = inference(tmp[0],batch_size) #4 inceptions
        print('len : ' + str(len(RN101Mcount)))
        latency2 = inference(RN101Mcount,batch_size)
        latency3 = inference(tmp[1],batch_size) #4 inceptions
        latency4 = inference(BBcount,batch_size)
        latency5 = inference(tmp[2],batch_size) #4 inceptions
        latency6 = inference(RN50Mcount,batch_size) 
        latency7 = inference(tmp[3],batch_size) #4 inceptions
        total_latency = latency1 + latency2 + latency3 + latency4 + latency5 + latency6 + latency7
        latency_list.append(total_latency)
        time.sleep(0.5)
    print('Latency for 50 runs requests : ' + str(latency_list))
    
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
    return latency
    #time.sleep(2)
    
if __name__ == '__main__':
    main()
