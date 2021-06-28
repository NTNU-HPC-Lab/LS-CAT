#include "includes.h"
__global__ void compute_d_vector_kernel(int N_i, int* d_ind, double* d_ptr, int* p_ptr, double* N_ptr, int N_ld) {
int I = threadIdx.x + blockIdx.x * blockDim.x;

if (I < N_i) {
int index = p_ptr[d_ind[I]];

d_ptr[d_ind[I]] = 1. / N_ptr[index + index * N_ld];
}
}