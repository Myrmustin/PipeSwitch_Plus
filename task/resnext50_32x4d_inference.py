import contextlib
import threading
import time

import torch
import numpy

import task.resnext50_32x4d as resnext50_32x4d
import task.common as util

TASK_NAME = 'resnext50_32x4d_inference'

@contextlib.contextmanager
def timer(prefix):
    _start = time.time()
    yield
    _end = time.time()
    print(prefix, 'cost', _end - _start)

def import_data_loader():
    return None

def import_model():
    model = resnext50_32x4d.import_model()
    model.eval()
    return model

def import_func():
    def inference(model, data_b):
        print(threading.currentThread().getName(),
            'resnext50_32x4d inference >>>>>>>>>>', time.time(),
            'model status', model.training)
        with timer('resnext50_32x4d inference func'):
            data = torch.from_numpy(numpy.frombuffer(data_b, dtype=numpy.float32))
            input_batch = data.view(-1, 3, 224, 224).cuda(non_blocking=True)
            with torch.no_grad():
                output = model(input_batch)
            return output.sum().item()
    return inference

def import_task():
    model = import_model()
    func = import_func()
    group_list = resnext50_32x4d.partition_model(model)
    shape_list = [util.group_to_shape(group) for group in group_list]
    return model, func, shape_list


def import_parameters():
    model = import_model()
    group_list = resnext50_32x4d.partition_model(model)
    batch_list = [util.group_to_batch(group) for group in group_list]
    return batch_list