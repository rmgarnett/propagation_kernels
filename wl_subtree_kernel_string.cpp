#include "mex.h"
#include "matrix.h"

#include <math.h>

#include <iostream>
using std::cout;
using std::endl;

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;

#include <tr1/functional>
#include <tr1/unordered_map>
using std::tr1::hash;
using std::tr1::unordered_map;

extern "C" {
#include <cblas.h>
}

/* convenience macros for input/output indices in case we want to change */
#define A_ARG               prhs[0]
#define LABELS_ARG          prhs[1]
#define GRAPH_IND_ARG       prhs[2]
#define H_ARG               prhs[3]

#define KERNEL_MATRIX_ARG   plhs[0]

#define INDEX(row, column, num_rows) ((int)(row) + ((int)(num_rows) * (int)(column)))

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{
	mwIndex *A_ir, *A_jc;
	double *graph_ind, *labels_in, *h_in, *kernel_matrix;
	int h, *labels;

	int i, j, k, row, column, count, offset, iteration, num_nodes, num_labels, num_new_labels, num_graphs,
		num_elements_this_column, index, *counts;

	double *feature_vectors;

	unordered_map<string, int, hash<string> > signature_hash;

	A_ir      = mxGetIr(A_ARG);
	A_jc      = mxGetJc(A_ARG);

	labels_in = mxGetPr(LABELS_ARG);

	graph_ind = mxGetPr(GRAPH_IND_ARG);
	h_in      = mxGetPr(H_ARG);

	/* dereference to avoid annoying casting and indexing */
	h = (int)(h_in[0] + 0.5);

	num_nodes = mxGetN(A_ARG);

	/* copy label matrix because we will overwrite it */
	labels = new int[num_nodes];
	for (i = 0; i < num_nodes; i++)
		labels[i] = (int)(labels_in[i] + 0.5);

	num_labels = 0;
	num_graphs = 0;
	for (i = 0; i < num_nodes; i++) {
		if (labels[i] > num_labels)
			num_labels = (int)(labels[i]);
		if ((int)(graph_ind[i]) > num_graphs)
			num_graphs = (int)(graph_ind[i] + 0.5);
	}

	KERNEL_MATRIX_ARG = mxCreateDoubleMatrix(num_graphs, num_graphs, mxREAL);
	kernel_matrix = mxGetPr(KERNEL_MATRIX_ARG);

	feature_vectors = NULL;
	counts          = NULL;

	iteration = 0;
	while (true) {

		delete[] feature_vectors;
		feature_vectors = new double[num_graphs * num_labels]();

		for (i = 0; i < num_nodes; i++)
			feature_vectors[INDEX(graph_ind[i] - 1, labels[i] - 1, num_graphs)]++;

		cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, num_graphs, num_labels,
		 						1.0, feature_vectors, num_graphs, 1.0, kernel_matrix, num_graphs);

		if (iteration == h)
			break;

		delete[] counts;
		counts = new int[num_nodes * num_labels];
		for (i = 0; i < num_nodes * num_labels; i++)
			counts[i] = 0;

 		count = 0;
		for (column = 0; column < num_nodes; column++) {
			num_elements_this_column = A_jc[column + 1] - A_jc[column];
			for (i = 0; i < num_elements_this_column; i++, count++) {
				row = A_ir[count];
				counts[INDEX(row, labels[column] - 1, num_nodes)]++;
			}
		}

		num_new_labels = 0;
		for (i = 0; i < num_nodes; i++) {
			ostringstream signature;
			signature << labels[i];

			for (j = 0; j < num_labels; j++)
				if (counts[INDEX(i, j, num_nodes)])
					signature << " " << j << " " << counts[INDEX(i, j, num_nodes)];

			if (signature_hash.count(signature.str()) == 0) {
				num_new_labels++;
				labels[i] = num_new_labels;
				signature_hash[signature.str()] = num_labels;
			}
			else
				labels[i] = signature_hash[signature.str()];
		}
		signature_hash.clear();

		num_labels = num_new_labels;
		iteration++;
	}

	delete[] labels;
	delete[] feature_vectors;
	delete[] counts;
}
