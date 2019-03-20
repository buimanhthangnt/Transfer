# -*- coding: utf-8 -*-
"""Thrift client module."""

import pickle
from gen_py.face_emb import FaceEmbedding
from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket
from thrift.transport import TTransport
import traceback
import config


# Make socket
def create_socket():
    transport = TSocket.TSocket('0.0.0.0', config.THRIFT_PORT)

    transport = TTransport.TBufferedTransport(transport)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = FaceEmbedding.Client(protocol)
    return client, transport


def _validate_client(transport):
    if not transport.isOpen():
        print("EmbClient not connected")
        transport.open()


def get_emb_numpy(array_of_numpy, client, transport):
    """
    Get embedded vectors from list of numpy array.

    array_of_numpy : list of image in numpy array format
    """
    send_data = []
    for array in array_of_numpy:
        pkl = pickle.dumps(array)
        send_data.append(pkl)
    result = []
    try:
        _validate_client(transport)
        result = client.get_emb_numpy(send_data)
    except:
        transport.close()
        print(traceback.print_exc())
        # print(str(exc))
    return result
