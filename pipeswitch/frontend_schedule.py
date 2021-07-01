import threading
import torch
import importlib

from util.util import timestamp

class FrontendScheduleThd(threading.Thread):
    def __init__(self, model_list, qin, worker_list):
        super(FrontendScheduleThd, self).__init__()
        self.model_list = model_list
        self.qin = qin
        self.worker_list = worker_list
        self.cur_w_idx = 0

    def run(self):
        timestamp('schedule', 'start')

        # Load models
        models = {}
        for model_name in self.model_list:
            models[hash(model_name)] = self._load_model(model_name)
        timestamp('schedule', 'load_model')

        # Create CUDA stream
        cuda_stream_for_parameter = torch.cuda.Stream()
        timestamp('schedule', 'create_stream')
        index =0 
        previous_request=''
        while True:
            index = index + 1
            print('Index is ' + str(index) ) 

            
            # Get request
            agent, model_name = self.qin.get()
            timestamp('schedule', 'get_request')
            print('Previous worker was on task: ' + previous_request + '. Curent request is: ' + model_name)
            if(previous_request == model_name):
                timestamp('schedule', 'REUSING WORKER')
                # Get current worker
                old_pipe, _, param_trans_pipe_parent,term_pipe = self.worker_list[self.cur_w_idx]
                timestamp('schedule', 'get_current_worker')
                # DONT Send terminate signal to current worker
                # DONT Get next worker to work on request
                # Send request to OLD worker
                old_pipe.send((agent, model_name))
                timestamp('schedule', 'notify_OLD_worker_on_NEW_TASK')

                # Transfer data to GPU OLD WORKER
                #data_b = self.qin.get()
                #old_pipe.send(data_b)
                #timestamp('schedule', 'send_data')

                # Allocate cache to streams
                with torch.cuda.stream(cuda_stream_for_parameter):
                    torch.cuda.insert_shared_cache_for_parameter() # pylint: disable=no-member
                timestamp('schedule', 'insert_cache')
                # Transfer parameters to GPU
                batched_parameter_list = models[hash(model_name)]
                self._transfer_parameter(old_pipe,
                                     batched_parameter_list, 
                                     cuda_stream_for_parameter,
                                     param_trans_pipe_parent)
                timestamp('schedule', 'transfer_parameters')
                # Clear status
                with torch.cuda.stream(cuda_stream_for_parameter):
                    torch.cuda.clear_shared_cache() # pylint: disable=no-member
                timestamp('schedule', 'clear_status')

                # Recv response
                res = old_pipe.recv()
                timestamp('schedule', 'get_response')
                previous_request = model_name
            else:
                # Get current worker
                model_list, pipe, param_trans_pipe,term_pipe = self.worker_list[self.cur_w_idx]
                timestamp('schedule', 'get_current_worker')
                # Send terminate signal to current worker
                term_pipe.send('terminate')

                # Get next worker to work on request
                self.cur_w_idx += 1
                self.cur_w_idx %= len(self.worker_list)
                new_pipe, _, param_trans_pipe_parent, _ = self.worker_list[self.cur_w_idx]

                # Send request to new worker
                new_pipe.send((agent, model_name))
                timestamp('schedule', 'notify_new_worker')

                # Wait for current worker to terminate
                resp = term_pipe.recv()
                timestamp('schedule', 'terminate_current_worker')

                # Transfer data to GPU
                data_b = self.qin.get()
                new_pipe.send(data_b)
                timestamp('schedule', 'send_data')

                # Allocate cache to streams
                with torch.cuda.stream(cuda_stream_for_parameter):
                    torch.cuda.insert_shared_cache_for_parameter() # pylint: disable=no-member
                timestamp('schedule', 'insert_cache')
                # Transfer parameters to GPU
                batched_parameter_list = models[hash(model_name)]
                self._transfer_parameter(new_pipe,
                                     batched_parameter_list, 
                                     cuda_stream_for_parameter,
                                     param_trans_pipe_parent)
                timestamp('schedule', 'transfer_parameters')

                # Clear status
                with torch.cuda.stream(cuda_stream_for_parameter):
                    torch.cuda.clear_shared_cache() # pylint: disable=no-member
                timestamp('schedule', 'clear_status')

                # Recv response
                ses = new_pipe.recv()
                previous_request = model_name
                timestamp('schedule', 'get_response')


    def _load_model(self, model_name):
        # Import parameters
        model_module = importlib.import_module('task.' + model_name)
        batched_parameter_list = model_module.import_parameters()

        # Preprocess batches
        processed_batched_parameter_list = []
        for param, mod_list in batched_parameter_list:
            if param is None:
                processed_batched_parameter_list.append((None, mod_list))
            else:
                processed_batched_parameter_list.append((param.pin_memory(), mod_list))

        return processed_batched_parameter_list

    def _transfer_parameter(self, pipe, 
                            batched_parameter_list, 
                            cuda_stream_for_parameter,
                            param_trans_pipe):
        param_cuda_list = []
        for param, mod_list in batched_parameter_list:
            with torch.cuda.stream(cuda_stream_for_parameter):
                if param is not None:
                    param_cuda = param.cuda(non_blocking=True)
                    param_cuda_list.append(param_cuda)
                    e = torch.cuda.Event()
                    e.record()
                    e.synchronize()
                param_trans_pipe.send(mod_list[0])
