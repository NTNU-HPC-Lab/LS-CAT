#include "includes.h"
__global__ void spmv_csr_vector_kernel(unsigned int computation_restriction_factor, const unsigned int* cum_row_indexes, const unsigned int* column_indexes, const float* matrix_data , const float* in_vector, float* out_vector, const unsigned int outerdim) {
__shared__ float vals[32];
int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
// global thread index
int warp_id = thread_id / 32;
// global warp index
int lane = thread_id & (32 - 1);
// thread index within the warp

int row = warp_id / computation_restriction_factor;
if (row < outerdim) {
int row_start = cum_row_indexes[row];
int row_end = cum_row_indexes[row+1];

// compute running prod per thread
vals[threadIdx.x] = 1;
for (int i = row_start + lane; i < row_end; i += 32) {
vals[threadIdx.x] *= 1 - (matrix_data[i] * in_vector[column_indexes[i]]);
}

// parallel reduction in shared memory
if (lane < 16) vals[threadIdx.x] *= vals[threadIdx.x + 16];
if (lane < 8) vals[threadIdx.x] *= vals[threadIdx.x + 8];
if (lane < 4) vals[threadIdx.x] *= vals[threadIdx.x + 4];
if (lane < 2) vals[threadIdx.x] *= vals[threadIdx.x + 2];
if (lane < 1) vals[threadIdx.x] *= vals[threadIdx.x + 1];

// first thread writes the result
if (lane == 0) out_vector[row] = vals[threadIdx.x];
}

}