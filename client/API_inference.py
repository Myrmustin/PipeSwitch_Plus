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
    # Load image
    #data = get_data(model_name, batch_size)
    print('List: ' + str(model_name_list))
    cur_data = ''
    latency_list = []
    for cur_model in model_name_list:
        timestamp('client', 'before_request')

        if(cur_data==''):
            data = get_data(cur_model, batch_size)
            cur_data = cur_model
        else:
            if(cur_data == cur_model):
                print("Using Same model")
            else:
                data = get_data(cur_model, batch_size)
                cur_data = cur_model
        # Connect
        client = TcpClient('localhost', 12345)
        timestamp('client', 'after_connect')
        time_1 = time.time()

        # Serialize data
        task_name = model_name + '_inference'
        task_name_b = task_name.encode()
        task_name_length = len(task_name_b)
        task_name_length_b = struct.pack('I', task_name_length)
        data_b = data.numpy().tobytes()
        length = len(data_b)
        length_b = struct.pack('I', length)
        timestamp('client', 'after_serialization')

        print("L: " + task_name_length)
        print("N: " + task_name)
        print("CurM: " + cur_model)
        print("CurD: " + cur_data)
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
        latency = (time_2 - time_1) * 1000
        print("Inference request on machine X using model " + cur_model + " (" + str(batch_size) + " batchsize) completed for: " + str(latency) + "ms. ")
        time.sleep(2)

    

if __name__ == '__main__':
    main()





'''def main():
    model_name = sys.argv[1]
    model_name_list = model_name.split(';')
    batch_size = int(sys.argv[2])

    cur_data = ""
    # Load image
    #data = get_data(model_name, batch_size)

    for model_name_o in model_name_list:
        timestamp('client', 'before_request')
        # Load image
        if(cur_data==''):
            data = get_data(model_name_o, batch_size)
            cur_data = model_name_o
        else:
            if(cur_data == model_name_o):
                print("Using Same model")
            else:
                data = get_data(model_name_o, batch_size)
                cur_data = model_name_o
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
        print("Inference request on machine X using model " + model_name_o + " (" + str(batch_size + " batchsize) completed for: " + str(latency) + "ms. ")
        

if __name__ == '__main__':
    main()
else:
    main()'''