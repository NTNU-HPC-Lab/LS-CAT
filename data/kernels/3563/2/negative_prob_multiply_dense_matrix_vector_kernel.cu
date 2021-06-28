#include "includes.h"
__global__ void negative_prob_multiply_dense_matrix_vector_kernel(float* matrix, float* in_vector, float* out_vector, unsigned int outerdim, unsigned int innerdim) {
// We parallelize at the level of matrix rows,
unsigned int row = blockIdx.x*blockDim.x+threadIdx.x;

float prob = 1.0;

if (row < outerdim) {
// each thread computes one element of the output vector
for (int i = 0; i < innerdim; i++) {
prob *= 1.0 - (matrix[row * innerdim + i] * in_vector[i]);
}
out_vector[row] = prob;
}
}