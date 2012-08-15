import numpy
from ctypes import *

lib = cdll.LoadLibrary('propagation_kernel.dylib')
lib.propagation_kernel.restype  = None
lib.propagation_kernel.argtypes = [POINTER(c_int),
                                   POINTER(c_double),
                                   c_int,
                                   c_int,
                                   c_int,
                                   c_double,
                                   c_int,
                                   POINTER(c_double)]

def compute(graph_ind, probabilities, w, p):

    num_graphs = int(graph_ind.max())
    num_nodes  = probabilities.shape[0]
    num_labels = probabilities.shape[1]

    K = numpy.zeros((num_graphs, num_graphs), float)

    graph_ind     = numpy.asfortranarray(graph_ind,     numpy.intc)
    probabilities = numpy.asfortranarray(probabilities, float)
    K             = numpy.asfortranarray(K)
    
    lib.propagation_kernel(graph_ind.ctypes.data_as(POINTER(c_int)),
                           probabilities.ctypes.data_as(POINTER(c_double)),
                           num_nodes,
                           num_labels,
                           num_graphs,
                           w,
                           int(p),
                           K.ctypes.data_as(POINTER(c_double)))
    
    return K
