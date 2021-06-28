#include "includes.h"
__global__ void revisedArraySum(float *array, float *sum){

__shared__ float partialSum[256];
int t = threadIdx.x;
for(int stride = 1;stride < blockDim.x; stride *= 2){
__syncthreads();
if(t % (2 * stride) == 0){
partialSum[t] += partialSum[t + stride];
}
}
sum[0] = partialSum[0];
}