import time

import torch
import torch.nn as nn

import task.resnext50_32x4d as resnext50_32x4d
import task.common as util

TASK_NAME = 'resnext50_32x4d_training'

def import_data_loader():
    return resnext50_32x4d.import_data

def import_model():
    model = resnext50_32x4d.import_model()
    model.train()
    return model

def import_func():
    def train(model, data_loader):
        # Prepare data
        batch_size = 32
        images, target = data_loader(batch_size)

        # Prepare training
        lr = 0.1
        momentum = 0.9
        weight_decay = 1e-4
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

        loss = None
        for i in range(100):
            # Data to GPU
            images_cuda = images.cuda(non_blocking=True)
            target_cuda = target.cuda(non_blocking=True)

            # compute output
            output = model(images_cuda)
            loss = criterion(output, target_cuda)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print ('Training', i, time.time(), loss.item())

            if i % 10 == 0:
                torch.save({'state_dict': model.state_dict()}, './saved_models/' + TASK_NAME.split('_')[0] + '.pth.tar')
                print('Model saved')

            del images_cuda
            del target_cuda
        
        return loss.item()
    return train

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