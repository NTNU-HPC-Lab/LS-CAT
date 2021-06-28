#include "includes.h"
__global__ void reduce_moments(float *d_arr, float *d_results, int N)
{
__shared__ float sh_array[pThreads];
int n = blockDim.x * blockIdx.x + threadIdx.x;
// sh_array[threadIdx.x] = 0;
if (n < N){
for (int s = blockDim.x / 2; s > 0; s >>= 1){
if ( threadIdx.x < s)
{
sh_array[threadIdx.x] += d_arr[threadIdx.x + s];
}
__syncthreads();
}

if (threadIdx.x ==0){
d_results[blockIdx.x] = sh_array[0];
// printf("%d %f\n", blockIdx.x, d_results[blockIdx.x]);
}
}
}