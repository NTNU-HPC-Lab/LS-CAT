#include "includes.h"
__global__ void gatherSum(int N, float *input, float *output){
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i >= N) return;
for(int j=0;j<N;++j){
output[i] += input[j];
}
return;
}