#include "includes.h"

extern "C" {
}


__device__ void softmax_device(int n, float *input, float temp, float *output)
{
int i;
float sum = 0;
float largest = -INFINITY;
for(i = 0; i < n; ++i){
int val = input[i];
largest = (val>largest) ? val : largest;
}
for(i = 0; i < n; ++i){
float e = exp(input[i]/temp - largest/temp);
sum += e;
output[i] = e;
}
for(i = 0; i < n; ++i){
output[i] /= sum;
}
}
__global__ void softmax_kernel(int n, int offset, int batch, float *input, float temp, float *output)
{
int b = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(b >= batch) return;
softmax_device(n, input + b*offset, temp, output + b*offset);
}