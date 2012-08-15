import numpy
import propagation_kernel

num_nodes   = 10
num_classes = 3

graph_ind = numpy.zeros((num_nodes, 1))
graph_ind[0:(graph_ind.shape[0] / 2)] = 1
graph_ind[(graph_ind.shape[0] / 2):]  = 2

probabilities = numpy.random.rand(num_nodes, num_classes)

w = 1e-4
p = 1

K = propagation_kernel.compute(graph_ind, probabilities, w, p)
