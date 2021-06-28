#include "includes.h"
__global__ void sub_calculation(    char* dev_a, char* dev_b, char* dev_c, int k, int j, int num_matrices, int matrix_size ) {
// Each thread handles a matrix
int i = (blockIdx.x*blockDim.x) + threadIdx.x;

if (i >= matrix_size) return;

int index = k*matrix_size*matrix_size+j*matrix_size+i;
dev_c[index] = dev_a[index] - dev_b[index];

}