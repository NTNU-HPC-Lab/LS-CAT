#include "includes.h"
__global__ void scatterSum(int N, float *input, float *output){
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i >= N) return;
float a = input[i];
for(int j=0;j<N;++j){
atomicAdd(output+(j+i)%N, a);
}
return;
}