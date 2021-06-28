#include "includes.h"
__global__ void scatterSum(int N, float *input, float *output){
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i >= N) return;
for(int j=0;j<N;++j){
atomicAdd(output+j, input[i]);
// if(i<N/2) atomicAdd(output+j, input[i]);
// atomicAdd(output+j, i<N/2: input[i]: 0.);
}
return;
}