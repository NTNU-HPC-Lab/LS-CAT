#include "includes.h"
__global__ void negative_prob_multiply_csr_matrix_vector_kernel(unsigned int* cum_row_indexes, unsigned int* column_indexes, float* matrix_data, float* in_vector, float* out_vector, unsigned int outerdim) {

unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;

if (row < outerdim) {
float prob = 1.0;

unsigned int row_start = cum_row_indexes[row];
unsigned int row_end = cum_row_indexes[row+1];

for (int i = row_start; i < row_end; i++) {
prob *= 1.0 - (matrix_data[i] * in_vector[column_indexes[i]]);
}
out_vector[row] = prob;
}
}