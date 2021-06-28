#include "includes.h"
__global__ void naiveKernel(int N, double *input, double *output){
int global_i = blockIdx.x * blockDim.x + threadIdx.x;
if(global_i < N){
for(int i=0;i<N;++i) output[global_i] += input[i];
output[global_i] /= N;
}
return ;
}