#include "includes.h"
__global__ void smemKernel(int N, float *input, float *output){
int b_size = blockDim.x, b_idx = blockIdx.x, t_idx = threadIdx.x;
int global_i = b_size * b_idx + t_idx, n_chk = (N + SHARE_SIZE - 1)/SHARE_SIZE;
__shared__ float buff[SHARE_SIZE];
for(int q=0;q<n_chk;++q){
int left = q*SHARE_SIZE, right = min(left + SHARE_SIZE, N);
for(int i = t_idx + left; i < right; i += b_size) buff[i-left] = input[i];
__syncthreads();
if(global_i < N){
for(int i = left; i < right; ++i) output[global_i] += buff[i-left];
}
__syncthreads();
}
output[global_i] /= N;
return ;
}