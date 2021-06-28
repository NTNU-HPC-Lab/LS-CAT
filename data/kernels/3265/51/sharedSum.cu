#include "includes.h"
__global__ void sharedSum(int N, float *input, float *output){
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i >= N) return;
__shared__ float tmp[BLOCK_SIZE];
memset(tmp, 0, sizeof(tmp));
float a = input[i];
for(int j=0;j<BLOCK_SIZE;++j){
atomicAdd(tmp + j, a);
}
__syncthreads();
output[blockDim.x*blockIdx.x + threadIdx.x] = tmp[threadIdx.x];
return;
}