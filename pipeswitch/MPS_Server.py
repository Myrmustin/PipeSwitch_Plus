from statistics import mode
import time

import sys
import queue
import struct
import threading
import importlib

import torch.multiprocessing as mp

from util.util import TcpServer, TcpAgent, timestamp

def func_get_request(qout):
    # Listen connections
    server = TcpServer('localhost', 12345)

    while True:
        # Get connection
        conn, _ = server.accept()
        agent = TcpAgent(conn)

        model_name_length_b = agent.recv(4)
        model_name_length = struct.unpack('I', model_name_length_b)[0]
        if model_name_length == 0:
            break
        model_name_b = agent.recv(model_name_length)
        model_name = model_name_b.decode()
        timestamp('tcp', 'get_name')

        data_length_b = agent.recv(4)
        data_length = struct.unpack('I', data_length_b)[0]
        if data_length > 0:
            data_b = agent.recv(data_length)
        else:
            data_b = None
        timestamp('tcp', 'get_data')
        qout.put((agent, model_name, data_b))

def func_schedule(qin):
    worker_list = []
    while True:
        agent, model_name, data_b = qin.get()
        isInference = model_name.find('_inference')
        print('model_name: ' + model_name + ". IsInference: " + str(isInference) )
        isTraining = model_name.find('_training')
        print('model_name: ' + model_name + ". IsTraining: " + str(isTraining) )
        if(isInference):
            active_worker = mp.Process(target=do_inference, args=(agent, model_name, data_b))
            active_worker.start()
            print('Started a worker to do inference!' )
            worker_list.append(active_worker)
        else:
            active_worker = mp.Process(target=do_training, args=(agent, model_name, data_b))
            active_worker.start()
            print('Started a worker to do training!' )
            worker_list.append(active_worker)
   
    
    
    """active_worker = None
    while True:
        agent, model_name, data_b = qin.get()
        if active_worker is not None:
            active_worker.kill()
            active_worker.join()
        active_worker = mp.Process(target=worker_compute, args=(agent, model_name, data_b))
        active_worker.start()"""

def do_training():
    return None
def do_inference():
    return None

def main():
    q_to_schedule = queue.Queue()
    t_get = threading.Thread(target=func_get_request, args=(q_to_schedule,))
    t_get.start()
    t_schedule = threading.Thread(target=func_schedule, args=(q_to_schedule,))
    t_schedule.start()

    # Accept connection
    t_get.join()
    t_schedule.join()
    

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()