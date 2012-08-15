#include <math.h>

#include <tr1/functional>
#include <tr1/unordered_map>
using std::tr1::hash;
using std::tr1::unordered_map;

#include <boost/random.hpp>
using boost::variate_generator;
using boost::normal_distribution;
using boost::cauchy_distribution;

typedef boost::mt19937 base_generator_type;

extern "C" {
#include <cblas.h>

#define INDEX(row, column, num_rows) ((int)(row) + ((int)(num_rows) * (int)(column)))
#define ROUND_TO_INT(x) ((int)((x) + 0.5))

void propagation_kernel(int *graph_ind, double *probabilities, int num_nodes,
												int num_labels, int num_graphs, double w, int p,
												double *kernel_matrix)
{
	int i, *labels;
	double *random_offsets, *signatures, *feature_vectors;

	unordered_map<double, int, hash<double> > signature_hash;

	base_generator_type generator(0);

	normal_distribution<double> normal(0, 1);
	variate_generator<base_generator_type&, normal_distribution<double> >
		randn(generator, normal);

	cauchy_distribution<double> cauchy(0, 1);
	variate_generator<base_generator_type&, cauchy_distribution<double> >
		randc(generator, cauchy);

	labels = new int[num_nodes];
	signatures = new double[num_nodes]();

	random_offsets = new double[num_labels - 1];
	for (i = 0; i < num_labels - 1; i++)
		if (p == 1)
			random_offsets[i] = randc();
		else
			random_offsets[i] = randn();

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, num_nodes, 1, num_labels - 1, 1,
							probabilities, num_nodes, random_offsets, num_labels - 1, 1, signatures, num_nodes);

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
	
}
