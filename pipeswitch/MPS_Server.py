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
      
        if '_inference' in model_name:
            active_worker = mp.Process(target=worker_compute, args=(agent, model_name, data_b))
            timestamp('INFERENCE IS STARTING', 'start')
            active_worker.start()
            timestamp('INFERENCE IS DONE', 'end')
            worker_list.append(active_worker)
        if '_training' in model_name:
            active_worker = mp.Process(target=worker_compute, args=(agent, model_name, data_b))
            active_worker.start()
            print('Started a worker to do training!' )
            worker_list.append(active_worker)
    #for i in worker_list:
       # i.join()

    
    
    """active_worker = None
    while True:
        agent, model_name, data_b = qin.get()
        if active_worker is not None:
            active_worker.kill()
            active_worker.join()
        active_worker = mp.Process(target=worker_compute, args=(agent, model_name, data_b))
        active_worker.start()"""

def worker_compute(agent, model_name, data_b):
    
    print('We skiped!')
    # Load model
    model_module = importlib.import_module('task.' + model_name)
    model, func, _ = model_module.import_task()
    data_loader = model_module.import_data_loader()

    # Model to GPU
    model = model.to('cuda')

    # Compute
    if 'training' in model_name:
        agent.send(b'FNSH')
        del agent
        timestamp('server', 'reply')

        output = func(model, data_loader)
        timestamp('server', 'complete')

    else:
        output = func(model, data_b)
        timestamp('server', 'complete')

        agent.send(b'FNSH')
        del agent
        timestamp('server', 'reply')

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
    #mp.set_start_method('spawn')
    main()