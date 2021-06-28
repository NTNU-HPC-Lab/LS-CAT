#include "includes.h"
__global__ void l2_norm_dVector_kernel(double *a, double *partial_sum, int n) {
__shared__ double partial_sums[BLOCKSIZE];

double local_sum = 0;

int id = blockIdx.x*blockDim.x + threadIdx.x;
int partial_index = threadIdx.x;

while (id < n) {
local_sum += (a[id] * a[id]);
id += (blockDim.x * gridDim.x); // this thread may have to handle multiple sums
}

partial_sums[partial_index] = local_sum;

__syncthreads();

int sum_level = blockDim.x >> 1; // divide by 2

while (sum_level != 0) {
if (partial_index < sum_level) {
partial_sums[partial_index] += partial_sums[partial_index + sum_level];
}

__syncthreads();

sum_level >>= 1; // divide by 2
}

if (partial_index == 0) {
// if we are the thread processing index 0 of partial_sums for our block
partial_sum[blockIdx.x] = partial_sums[0];
}
// at this point there is still some partial somes left to compute
// inefficient to do so on GPU. Let CPU do this
}