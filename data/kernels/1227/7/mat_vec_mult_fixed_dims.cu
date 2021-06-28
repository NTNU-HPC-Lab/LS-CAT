#include "includes.h"
__global__ void mat_vec_mult_fixed_dims(int *mat, int *vec, int *res) {
int mat_rows = 1024;
int mat_cols = 512;
// El for each thread, shared per block
__shared__ int smem[128];
for (int block_i = 0; block_i * gridDim.x < mat_rows; block_i++) {
int row = blockIdx.x + (block_i * gridDim.x);
int row_total = 0;
for (int thread_i = 0; thread_i * blockDim.x < mat_cols; thread_i++) {
int col = threadIdx.x + (thread_i * blockDim.x);
// Load mult in shmem
smem[threadIdx.x] = mat[row * mat_cols + col] * vec[col];
__syncthreads();

// Parallel reduction
for (int i = blockDim.x / 2; i > 0; i /= 2) {
if (threadIdx.x < i) {
int temp = smem[threadIdx.x] + smem[threadIdx.x + i];
smem[threadIdx.x] = temp;
}
__syncthreads();
}
// Only 1 thread needs to do this
if (threadIdx.x == 0)
row_total += smem[threadIdx.x];
}
// Load into ans (single thread)
if (threadIdx.x == 0)
res[row] = row_total;
}
}