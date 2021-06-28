#include "includes.h"
__device__ void softmax_device(float *input, int n, float temp, int stride, float *output)
{
int i;
float sum = 0;
float largest = -INFINITY;
for(i = 0; i < n; ++i){
int val = input[i*stride];
largest = (val>largest) ? val : largest;
}
for(i = 0; i < n; ++i){
float e = expf(input[i*stride]/temp - largest/temp);
sum += e;
output[i*stride] = e;
}
for(i = 0; i < n; ++i){
output[i*stride] /= sum;
}
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
__global__ void softmax_kernel(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (id >= batch*groups) return;
int b = id / groups;
int g = id % groups;
softmax_device(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
}