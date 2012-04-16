#include "mex.h"
#include "matrix.h"

#include <math.h>

#include <iostream>
using std::cout;
using std::endl;

#include <tr1/functional>
#include <tr1/unordered_map>
using std::tr1::hash;
using std::tr1::unordered_map;

#include <boost/random.hpp>
using boost::uniform_real;
using boost::normal_distribution;
using boost::variate_generator;

typedef boost::mt19937 base_generator_type;

extern "C" {
#include <cblas.h>
}

/* convenience macros for input/output indices in case we want to change */
#define GRAPH_IND_ARG      prhs[0]
#define PROBABILITIES_ARG  prhs[1]
#define W_ARG              prhs[2]
#define NUM_VECTORS_ARG    prhs[3]
#define SQRT_FLAG_ARG      prhs[4]

#define KERNEL_MATRIX_ARG  plhs[0]

#define INDEX(row, column, num_rows) ((int)(row) + ((int)(num_rows) * (int)(column)))

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{
	double *graph_ind, *probabilities, *w_in, *num_vectors_in, *kernel_matrix;
	mxLogical *sqrt_flag_in;

	double w;
	int num_vectors;
	bool sqrt_flag;

	int i, j, count, num_nodes, num_labels, num_graphs, *labels;
	float *random_offsets, *signatures;
	double *feature_vectors, b;

	unordered_map<float, int, hash<float> > signature_hash;

	base_generator_type generator(0);

	uniform_real<float> uniform(0, 1);
	variate_generator<base_generator_type&, uniform_real<float> >
		rand(generator, uniform);

	normal_distribution<float> normal(0, 1);
	variate_generator<base_generator_type&, normal_distribution<float> >
		randn(generator, normal);

	graph_ind      = mxGetPr(GRAPH_IND_ARG);
	probabilities  = mxGetPr(PROBABILITIES_ARG);
	w_in           = mxGetPr(W_ARG);
	num_vectors_in = mxGetPr(NUM_VECTORS_ARG);
	sqrt_flag_in   = mxGetLogicals(SQRT_FLAG_ARG);

	/* dereference to avoid annoying casting and indexing */
	w           = w_in[0];
	sqrt_flag   = sqrt_flag_in[0];
	num_vectors = (int)(num_vectors_in[0] + 0.5);

	num_nodes  = mxGetM(PROBABILITIES_ARG);
	num_labels = mxGetN(PROBABILITIES_ARG);

	labels = new int[num_nodes]();

	num_graphs = 0;
	for (i = 0; i < num_nodes; i++) {
		if ((int)(graph_ind[i] + 0.5) > num_graphs)
			num_graphs = (int)(graph_ind[i] + 0.5);
	}

	KERNEL_MATRIX_ARG = mxCreateDoubleMatrix(num_graphs, num_graphs, mxREAL);
	kernel_matrix = mxGetPr(KERNEL_MATRIX_ARG);

	signatures = new float[num_nodes];

	random_offsets = new float[(num_labels - 1) * num_vectors];
	for (i = 0; i < (num_labels - 1) * num_vectors; i++)
		random_offsets[i] = randn();

	b = rand() * w;

	for (i = 0; i < num_nodes; i++)
		signatures[i] = b;

	count = 0;
	for (j = 0; j < num_labels - 1; j++) {
		for (i = 0; i < num_nodes; i++, count++)
			if (sqrt_flag)
				signatures[i] += sqrt(probabilities[count]) * random_offsets[j];
			else
				signatures[i] +=      probabilities[count]  * random_offsets[j];
	}

	for (i = 0; i < num_nodes; i++)
		signatures[i] = floor(signatures[i] / w);

	num_labels = 0;
	for (i = 0; i < num_nodes; i++) {
		if (signature_hash.count(signatures[i]) == 0) {
			num_labels++;
			labels[i] = num_labels;
			signature_hash[signatures[i]] = num_labels;
		}
		else
			labels[i] = signature_hash[signatures[i]];
	}

	feature_vectors = new double[num_graphs * num_labels]();
	for (i = 0; i < num_nodes; i++)
		feature_vectors[INDEX(graph_ind[i] - 1, labels[i] - 1, num_graphs)]++;

	cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, num_graphs, num_labels,
							1.0, feature_vectors, num_graphs, 0.0, kernel_matrix, num_graphs);

	delete[] labels;
	delete[] feature_vectors;
	delete[] random_offsets;
	delete[] signatures;
}
