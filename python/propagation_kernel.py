import numpy
from ctypes import *

lib = cdll.LoadLibrary('propagation_kernel.so')
lib.propagation_kernel.restype  = None
lib.propagation_kernel.argtypes = [POINTER(c_double),
                                   POINTER(c_int),
                                   c_double,
                                   c_int,
                                   c_int,
                                   c_int,
                                   c_int,
                                   POINTER(c_double)]

def compute(graph_ind, probabilities, w, p):

    num_graphs = int(graph_ind.max())
    num_nodes  = probabilities.shape[0]
    num_classes = probabilities.shape[1]

    K = numpy.zeros((num_graphs, num_graphs), float)

    graph_ind     = numpy.asfortranarray(graph_ind,     numpy.intc)
    probabilities = numpy.asfortranarray(probabilities, float)
    K             = numpy.asfortranarray(K)

    lib.calculate_propagation_kernel_contribution(probabilities.ctypes.data_as(POINTER(c_double)),
                                                  graph_ind.ctypes.data_as(POINTER(c_int)),
                                                  w,
                                                  int(p),
                                                  num_nodes,
                                                  num_classes,
                                                  num_graphs,
                                                  K.ctypes.data_as(POINTER(c_double)))

    return K
