import threading
import struct

from util.util import timestamp


class FrontendTcpThd(threading.Thread):
    def __init__(self, qout, agent):
        super(FrontendTcpThd, self).__init__()
        self.qout = qout
        self.agent = agent

    def run(self):
        while True:
            timestamp('tcp', 'listening')
            #print('First point of contact is in frontend_tcp.py')
            type_len_b =  self.agent.recv(4)
            print("type_len_b AFTER send: " + str(type_len_b))
   
            type_len = struct.unpack('I', type_len_b)[0]
            print("type_len AFTER unpack: " + str(type_len))

            type_b = self.agent.recv(type_len)
            print("type_b AFTER send: " + str(type_b))

            type = type_b.decode()
            print("Type AFTER decode is : " + type)
            str = self.regularFrontEndTcp()
            if(str == "stop"):
                break

            
    def regularFrontEndTcp(self):
        model_name_length_b = self.agent.recv(4)
        print("Recieved model name length in bytes:" + str(model_name_length_b) )
        model_name_length = struct.unpack('I', model_name_length_b)[0]
        print("Pos 1")

        if model_name_length == 0:
            #break
            return "stop"
        model_name_b = self.agent.recv(model_name_length)
        model_name = model_name_b.decode()

        self.qout.put((self.agent, model_name))
        timestamp('tcp', 'get_name')

        data_length_b = self.agent.recv(4)
        data_length = struct.unpack('I', data_length_b)[0]

        if data_length > 0:
            data_b = self.agent.recv(data_length)
        else:
            data_b = None
        timestamp('tcp', 'get_data')
        self.qout.put(data_b)
        timestamp('tcp', 'enqueue_request')
        return "OK"

        def modifiedFrontEndTcp():
            return None